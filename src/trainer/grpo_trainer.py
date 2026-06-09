"""
GRPO (Generalized Reinforced Policy Optimization) 训练器实现

基于 DeepSeekMath 论文:
[3] Shao et al., "DeepSeekMath: Pushing the Limits of Mathematical Reasoning 
    in Open Language Models", arXiv:2402.03300, 2024.

GRPO 特点:
1. 不需要训练单独的 Value Function
2. 使用 Group Relative Policy Optimization
3. 对每个问题采样多个回答，计算组内相对优势
4. 使用 clip 约束来限制策略更新幅度
"""
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
from tqdm import tqdm
import logging

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_scheduler
)
from peft import LoraConfig, get_peft_model, TaskType
from accelerate import Accelerator

from data_converter import RLDataItem, RewardFunction
from ppo_trainer import PPOConfig, RLDataset


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class GRPOConfig:
    """GRPO 训练配置"""
    
    # 模型配置
    model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    
    # LoRA 配置
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = None
    
    # 训练配置
    learning_rate: float = 1e-5  # GRPO 通常使用更小的学习率
    batch_size: int = 4
    num_epochs: int = 3
    max_seq_length: int = 2048
    max_prompt_length: int = 1024
    max_response_length: int = 512
    
    # GRPO 特定配置
    clip_epsilon: float = 0.2          # GRPO clipping epsilon
    group_size: int = 4                # 每个问题的采样数量 (G in paper)
    kl_coef: float = 0.04              # KL 散度惩罚系数 (通常比 PPO 小)
    
    # GRPO epoch 配置
    grpo_epochs: int = 2             # 每次采样的 epoch 数
    
    # 优化器配置
    warmup_steps: int = 100
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    
    # 生成配置
    temperature: float = 0.8
    top_p: float = 0.9
    top_k: int = 50
    do_sample: bool = True
    
    # 输出配置
    output_dir: str = "outputs/grpo"
    save_steps: int = 500
    logging_steps: int = 10
    
    # 其他
    seed: int = 42
    
    # Online filter (Bonus 题目相关)
    use_online_filter: bool = False
    advantage_var_low: float = 0.01
    advantage_var_high: float = 1.0


class GRPOTrainer:
    """
    GRPO 训练器
    
    与 PPO 的主要区别:
    1. 不使用 Value Function，而是对每个问题采样 G 个回答
    2. 计算组内相对优势: A_i = (r_i - mean(r)) / std(r)
    3. 使用与 PPO 类似的 clip 约束
    
    算法流程:
    1. 对于每个 prompt，采样 G 个 response
    2. 计算每个 response 的奖励 r_i
    3. 计算组内相对优势: A_i = (r_i - mean(r)) / std(r)
    4. 使用 PPO-clip 目标函数更新策略
    """
    
    def __init__(
        self,
        config: GRPOConfig,
        train_dataset: List[RLDataItem],
        accelerator: Optional[Accelerator] = None
    ):
        self.config = config
        self.accelerator = accelerator or Accelerator()
        
        # 设置随机种子
        torch.manual_seed(config.seed)
        
        # 初始化奖励函数
        self.reward_fn = RewardFunction()
        
        # 加载 tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name,
            trust_remote_code=True,
            padding_side="left"
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 加载策略模型
        logger.info(f"加载策略模型: {config.model_name}")
        self.policy_model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True
        )
        
        # 应用 LoRA
        if config.use_lora:
            self._apply_lora()
        
        # 加载参考模型（冻结参数）
        logger.info("加载参考模型...")
        self.ref_model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True
        )
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False
        
        # 准备优化器
        self.optimizer = self._create_optimizer()
        
        # 准备数据集
        self.train_dataset = RLDataset(train_dataset)
        # 注意：GRPO 使用不同的 batch 处理，这里 batch_size 是问题数量
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=self._collate_fn
        )
        
        # 准备学习率调度器
        num_training_steps = (
            len(self.train_dataloader) 
            * config.num_epochs 
        )
        self.lr_scheduler = get_scheduler(
            "cosine",
            optimizer=self.optimizer,
            num_warmup_steps=config.warmup_steps,
            num_training_steps=num_training_steps
        )
        
        # 使用 accelerator 准备
        if self.accelerator:
            self.policy_model, self.ref_model, self.optimizer, \
            self.train_dataloader, self.lr_scheduler = \
                self.accelerator.prepare(
                    self.policy_model, self.ref_model, self.optimizer,
                    self.train_dataloader, self.lr_scheduler
                )
        
        # 训练状态
        self.global_step = 0
        self.epoch = 0
        
        # 创建输出目录
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    
    def _apply_lora(self):
        """应用 LoRA 到策略模型"""
        if self.config.target_modules is None:
            target_modules = []
            for name, module in self.policy_model.named_modules():
                if isinstance(module, nn.Linear) and any(x in name for x in ["q_proj", "k_proj", "v_proj", "o_proj"]):
                    target_modules.append(name.split(".")[-1])
            self.config.target_modules = list(set(target_modules))
        
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.target_modules
        )
        
        self.policy_model = get_peft_model(self.policy_model, lora_config)
        logger.info(f"LoRA 配置: r={self.config.lora_r}, alpha={self.config.lora_alpha}")
        self.policy_model.print_trainable_parameters()
    
    def _create_optimizer(self):
        """创建优化器"""
        trainable_params = [p for p in self.policy_model.parameters() if p.requires_grad]
        
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=(0.9, 0.999)
        )
        return optimizer
    
    def _collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """数据批处理"""
        prompt_ids = [item["prompt_ids"] for item in batch]
        max_prompt_len = max(len(p) for p in prompt_ids)
        
        padded_prompts = []
        attention_masks = []
        for prompt in prompt_ids:
            padding_length = max_prompt_len - len(prompt)
            padded_prompt = [self.tokenizer.pad_token_id] * padding_length + prompt
            padded_prompts.append(padded_prompt)
            attention_masks.append([0] * padding_length + [1] * len(prompt))
        
        return {
            "ids": [item["id"] for item in batch],
            "prompts": [item["prompt"] for item in batch],
            "prompt_ids": torch.tensor(padded_prompts, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            "reference_answers": [item["reference_answer"] for item in batch],
            "events": [item["events"] for item in batch],
            "contexts": [item["context"] for item in batch]
        }
    
    @torch.no_grad()
    def generate_group_responses(
        self,
        prompt_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        为每个 prompt 生成 G 个 response (组采样)
        
        Args:
            prompt_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
        
        Returns:
            group_response_ids: [batch_size * G, response_len]
            group_response_mask: [batch_size * G, response_len]
        """
        batch_size = prompt_ids.shape[0]
        G = self.config.group_size
        
        # 重复 prompt G 次
        # [batch_size, seq_len] -> [batch_size * G, seq_len]
        repeated_prompt_ids = prompt_ids.unsqueeze(1).repeat(1, G, 1).view(-1, prompt_ids.shape[1])
        repeated_attention_mask = attention_mask.unsqueeze(1).repeat(1, G, 1).view(-1, attention_mask.shape[1])
        
        # 生成 response
        outputs = self.policy_model.generate(
            input_ids=repeated_prompt_ids,
            attention_mask=repeated_attention_mask,
            max_new_tokens=self.config.max_response_length,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            top_k=self.config.top_k,
            do_sample=self.config.do_sample,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            return_dict_in_generate=True
        )
        
        # 分离 response
        response_ids = outputs.sequences[:, prompt_ids.shape[1]:]
        response_mask = (response_ids != self.tokenizer.pad_token_id).long()
        
        return response_ids, response_mask
    
    @torch.no_grad()
    def compute_group_rewards(
        self,
        response_ids: torch.Tensor,
        events: List[List[Dict]],
        contexts: List[str]
    ) -> torch.Tensor:
        """
        计算组内每个 response 的奖励
        
        Args:
            response_ids: [batch_size * G, response_len]
            events: List of event lists (batch_size)
            contexts: List of contexts (batch_size)
        
        Returns:
            rewards: [batch_size, G] 每个问题的 G 个奖励
        """
        batch_size = len(events)
        G = self.config.group_size
        
        rewards = []
        
        for i in range(batch_size):
            group_rewards = []
            for g in range(G):
                idx = i * G + g
                response_text = self.tokenizer.decode(
                    response_ids[idx], skip_special_tokens=True
                )
                
                reward_dict = self.reward_fn.compute_reward(
                    response_text,
                    events[i]
                )
                group_rewards.append(reward_dict["total"])
            
            rewards.append(group_rewards)
        
        return torch.tensor(rewards, dtype=torch.float32).to(response_ids.device)
    
    def compute_group_relative_advantage(
        self,
        rewards: torch.Tensor
    ) -> torch.Tensor:
        """
        计算组内相对优势
        
        A_i = (r_i - mean(r)) / std(r)
        
        Args:
            rewards: [batch_size, G] 每个问题的 G 个奖励
        
        Returns:
            advantages: [batch_size, G] 组内相对优势
        """
        # 计算每个组的均值和标准差
        mean_rewards = rewards.mean(dim=1, keepdim=True)  # [batch_size, 1]
        std_rewards = rewards.std(dim=1, keepdim=True)    # [batch_size, 1]
        
        # 避免除零
        std_rewards = torch.clamp(std_rewards, min=1e-8)
        
        # 计算相对优势
        advantages = (rewards - mean_rewards) / std_rewards  # [batch_size, G]
        
        return advantages
    
    def filter_batch_by_advantage_variance(
        self,
        advantages: torch.Tensor,
        prompt_ids: torch.Tensor,
        response_ids: torch.Tensor,
        response_mask: torch.Tensor,
        old_log_probs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Online Filter: 根据优势方差筛选 batch
        
        Bonus 题目 (5a): 只保留优势方差在指定区间内的样本
        
        Args:
            advantages: [batch_size, G]
            ...
        
        Returns:
            筛选后的 tensors
        """
        if not self.config.use_online_filter:
            return prompt_ids, response_ids, response_mask, old_log_probs
        
        # 计算每个问题的优势方差
        advantage_var = advantages.var(dim=1)  # [batch_size]
        
        # 筛选条件
        valid_mask = (
            (advantage_var >= self.config.advantage_var_low) &
            (advantage_var <= self.config.advantage_var_high)
        )
        
        valid_indices = torch.where(valid_mask)[0]
        
        if len(valid_indices) == 0:
            logger.warning("所有样本都被过滤，保留所有样本")
            return prompt_ids, response_ids, response_mask, old_log_probs
        
        # 筛选
        valid_prompt_ids = prompt_ids[valid_indices]
        
        # response 需要特殊处理（每个 prompt 有 G 个 response）
        valid_response_ids = []
        valid_response_masks = []
        valid_old_log_probs = []
        
        for idx in valid_indices:
            for g in range(self.config.group_size):
                response_idx = idx * self.config.group_size + g
                valid_response_ids.append(response_ids[response_idx])
                valid_response_masks.append(response_mask[response_idx])
                valid_old_log_probs.append(old_log_probs[response_idx])
        
        return (
            valid_prompt_ids,
            torch.stack(valid_response_ids),
            torch.stack(valid_response_masks),
            torch.stack(valid_old_log_probs)
        )
    
    @torch.no_grad()
    def compute_logprobs(
        self,
        model,
        prompt_ids: torch.Tensor,
        response_ids: torch.Tensor,
        response_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        计算 log probabilities
        
        Args:
            model: 模型
            prompt_ids: [batch_size, prompt_len]
            response_ids: [batch_size * G, response_len]
            response_mask: [batch_size * G, response_len]
        
        Returns:
            log_probs: [batch_size * G, response_len]
        """
        batch_size = prompt_ids.shape[0]
        G = self.config.group_size
        
        # 构建完整序列
        full_ids = torch.cat([prompt_ids.repeat_interleave(G, dim=0), response_ids], dim=1)
        full_mask = torch.cat([
            torch.ones(batch_size * G, prompt_ids.shape[1], device=prompt_ids.device),
            response_mask
        ], dim=1)
        
        outputs = model(
            input_ids=full_ids,
            attention_mask=full_mask,
            return_dict=True
        )
        
        # 计算 log probs
        logits = outputs.logits[:, :-1, :]
        log_probs = F.log_softmax(logits, dim=-1)
        
        # 获取实际 token 的 log prob
        target_ids = full_ids[:, 1:]
        token_log_probs = log_probs.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)
        
        # 只取 response 部分
        response_len = response_ids.shape[1]
        response_log_probs = token_log_probs[:, -response_len:]
        
        # 应用 mask
        response_log_probs = response_log_probs * response_mask.float()
        
        return response_log_probs
    
    def grpo_loss(
        self,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        ref_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        response_mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        计算 GRPO 损失
        
        L_GRPO(θ) = E[ min(ratio * A, clip(ratio, 1-ε, 1+ε) * A) ] - β * KL(π_θ || π_ref)
        
        其中:
        - ratio = π_θ(a|s) / π_θ_old(a|s)
        - KL = log(π_θ / π_ref) - log(π_θ_old / π_ref) + log(π_θ_old / π_θ)
        
        Args:
            log_probs: [batch_size * G, response_len]
            old_log_probs: [batch_size * G, response_len]
            ref_log_probs: [batch_size * G, response_len]
            advantages: [batch_size, G] 需要扩展为 [batch_size * G, 1]
            response_mask: [batch_size * G, response_len]
        
        Returns:
            loss_dict: 包含各项损失
        """
        batch_size = advantages.shape[0]
        G = self.config.group_size
        
        # 扩展 advantages 到每个 token
        # [batch_size, G] -> [batch_size * G, 1, 1]
        expanded_advantages = advantages.view(-1, 1, 1)  # [batch_size * G, 1, 1]
        
        # 计算 ratio
        ratio = torch.exp(log_probs - old_log_probs)
        
        # 计算 surrogate
        surrogate1 = ratio * expanded_advantages
        surrogate2 = torch.clamp(
            ratio,
            1 - self.config.clip_epsilon,
            1 + self.config.clip_epsilon
        ) * expanded_advantages
        
        # PPO-clip 损失 (取最小)
        policy_loss = -torch.min(surrogate1, surrogate2)
        policy_loss = (policy_loss * response_mask.unsqueeze(-1)).sum() / response_mask.sum()
        
        # 计算 KL 散度 (使用估计器)
        # KL(π_θ || π_ref) ≈ log(π_θ) - log(π_ref)
        kl_penalty = (log_probs - ref_log_probs)
        kl_penalty = (kl_penalty * response_mask.unsqueeze(-1)).sum() / response_mask.sum()
        
        # 总损失
        total_loss = policy_loss + self.config.kl_coef * kl_penalty
        
        return {
            "total_loss": total_loss,
            "policy_loss": policy_loss,
            "kl_penalty": kl_penalty,
            "mean_ratio": ratio.mean(),
            "mean_advantage": advantages.mean()
        }
    
    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """
        执行一个 GRPO 训练步骤
        """
        prompt_ids = batch["prompt_ids"]
        attention_mask = batch["attention_mask"]
        batch_size = prompt_ids.shape[0]
        
        # 1. 采样阶段: 为每个 prompt 生成 G 个 response
        with torch.no_grad():
            response_ids, response_mask = self.generate_group_responses(
                prompt_ids, attention_mask
            )
            
            # 计算奖励 [batch_size, G]
            rewards = self.compute_group_rewards(
                response_ids,
                batch["events"],
                batch["contexts"]
            )
            
            # 计算组内相对优势
            advantages = self.compute_group_relative_advantage(rewards)
        
        # 2. Online Filter (如果启用)
        if self.config.use_online_filter:
            prompt_ids, response_ids, response_mask, _ = \
                self.filter_batch_by_advantage_variance(
                    advantages, prompt_ids, response_ids, response_mask, None
                )
            # 重新计算 rewards 和 advantages
            with torch.no_grad():
                rewards = self.compute_group_rewards(
                    response_ids,
                    batch["events"],
                    batch["contexts"]
                )
                advantages = self.compute_group_relative_advantage(rewards)
        
        # 3. 计算 old_log_probs (参考模型)
        with torch.no_grad():
            old_log_probs = self.compute_logprobs(
                self.ref_model, prompt_ids, response_ids, response_mask
            )
            
            # 计算 ref_log_probs (参考模型，用于 KL)
            ref_log_probs = old_log_probs.clone()
        
        # 4. GRPO 更新阶段
        total_losses = []
        policy_losses = []
        kl_penalties = []
        
        for grpo_epoch in range(self.config.grpo_epochs):
            # 计算当前策略的 log probs
            log_probs = self.compute_logprobs(
                self.policy_model, prompt_ids, response_ids, response_mask
            )
            
            # 计算 GRPO 损失
            loss_dict = self.grpo_loss(
                log_probs,
                old_log_probs,
                ref_log_probs,
                advantages,
                response_mask
            )
            
            # 反向传播
            self.optimizer.zero_grad()
            if self.accelerator:
                self.accelerator.backward(loss_dict["total_loss"])
                self.accelerator.clip_grad_norm_(
                    self.policy_model.parameters(),
                    self.config.max_grad_norm
                )
            else:
                loss_dict["total_loss"].backward()
                torch.nn.utils.clip_grad_norm_(
                    self.policy_model.parameters(),
                    self.config.max_grad_norm
                )
            
            self.optimizer.step()
            self.lr_scheduler.step()
            
            total_losses.append(loss_dict["total_loss"].item())
            policy_losses.append(loss_dict["policy_loss"].item())
            kl_penalties.append(loss_dict["kl_penalty"].item())
        
        return {
            "total_loss": sum(total_losses) / len(total_losses),
            "policy_loss": sum(policy_losses) / len(policy_losses),
            "kl_penalty": sum(kl_penalties) / len(kl_penalties),
            "mean_reward": rewards.mean().item(),
            "std_reward": rewards.std().item(),
            "mean_advantage": advantages.mean().item(),
            "std_advantage": advantages.std().item()
        }
    
    def train(self):
        """
        主训练循环
        """
        logger.info("开始 GRPO 训练...")
        
        for epoch in range(self.config.num_epochs):
            self.epoch = epoch
            logger.info(f"Epoch {epoch + 1}/{self.config.num_epochs}")
            
            progress_bar = tqdm(
                self.train_dataloader,
                desc=f"Epoch {epoch + 1}",
                disable=not self.accelerator.is_main_process if self.accelerator else False
            )
            
            for step, batch in enumerate(progress_bar):
                metrics = self.train_step(batch)
                self.global_step += 1
                
                # 更新进度条
                progress_bar.set_postfix(metrics)
                
                # 日志记录
                if self.global_step % self.config.logging_steps == 0:
                    logger.info(
                        f"Step {self.global_step}: "
                        f"loss={metrics['total_loss']:.4f}, "
                        f"reward={metrics['mean_reward']:.4f}±{metrics['std_reward']:.4f}, "
                        f"adv={metrics['mean_advantage']:.4f}±{metrics['std_advantage']:.4f}"
                    )
                
                # 保存检查点
                if self.global_step % self.config.save_steps == 0:
                    self.save_checkpoint()
            
            # 每个 epoch 结束保存
            self.save_checkpoint(suffix=f"epoch_{epoch + 1}")
        
        logger.info("训练完成!")
        self.save_checkpoint(suffix="final")
    
    def save_checkpoint(self, suffix: str = ""):
        """
        保存模型检查点
        """
        if self.accelerator and not self.accelerator.is_main_process:
            return
        
        save_dir = Path(self.config.output_dir)
        if suffix:
            save_dir = save_dir / f"checkpoint-{suffix}"
        else:
            save_dir = save_dir / f"checkpoint-{self.global_step}"
        
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存模型
        if self.config.use_lora:
            self.policy_model.save_pretrained(save_dir)
        else:
            self.policy_model.save_pretrained(save_dir)
        
        self.tokenizer.save_pretrained(save_dir)
        
        # 保存配置
        with open(save_dir / "training_config.json", 'w') as f:
            json.dump(self.config.__dict__, f, indent=2)
        
        logger.info(f"检查点已保存到: {save_dir}")


def main():
    """
    示例用法
    """
    from data_download import generate_timeline_data
    from data_converter import SFTtoRLConverter
    
    # 生成数据
    print("生成数据...")
    sft_data = generate_timeline_data(num_samples=100)
    
    # 转换为 RL 格式
    print("转换数据格式...")
    converter = SFTtoRLConverter()
    rl_data = converter.convert_sft_to_rl(sft_data)
    
    # 创建配置
    config = GRPOConfig(
        num_epochs=1,
        batch_size=2,
        group_size=4,
        output_dir="outputs/grpo_test"
    )
    
    # 创建训练器并训练
    print("开始 GRPO 训练...")
    trainer = GRPOTrainer(config, rl_data)
    trainer.train()


if __name__ == "__main__":
    main()
