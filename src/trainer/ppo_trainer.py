"""
PPO (Proximal Policy Optimization) 训练器实现
基于 TRL 库实现，适配 Qwen-2.5-VL-3B-Instruct 模型
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
    get_scheduler,
    TrainerCallback
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from accelerate import Accelerator

from data_converter import RLDataItem, RewardFunction


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PPOConfig:
    """PPO 训练配置"""
    
    # 模型配置
    model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    
    # LoRA 配置
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = None
    
    # 训练配置
    learning_rate: float = 5e-5
    batch_size: int = 4
    mini_batch_size: int = 1
    num_epochs: int = 3
    max_seq_length: int = 2048
    max_prompt_length: int = 1024
    max_response_length: int = 512
    
    # PPO 特定配置
    clip_epsilon: float = 0.2          # PPO clipping epsilon
    value_clip: float = 0.2            # Value function clipping
    ppo_epochs: int = 4                # PPO 每次更新的 epoch 数
    kl_coef: float = 0.2               # KL 散度惩罚系数
    gamma: float = 1.0                 # 折扣因子
    lam: float = 0.95                  # GAE lambda
    
    # 优化器配置
    warmup_steps: int = 100
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    
    # 生成配置
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    do_sample: bool = True
    
    # 输出配置
    output_dir: str = "outputs/ppo"
    save_steps: int = 500
    logging_steps: int = 10
    
    # 其他
    seed: int = 42
    device: str = "auto"


class RLDataset(Dataset):
    """RL 训练数据集"""
    
    def __init__(self, data: List[RLDataItem]):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx) -> Dict[str, Any]:
        item = self.data[idx]
        return {
            "id": item.id,
            "prompt": item.prompt,
            "prompt_ids": item.prompt_ids,
            "reference_answer": item.reference_answer,
            "reference_ids": item.reference_ids if item.reference_ids else [],
            "events": item.events,
            "context": item.context
        }


class PPOTrainer:
    """
    PPO 训练器
    
    基于以下论文实现:
    [1] Schulman et al., "Proximal Policy Optimization Algorithms", arXiv:1707.06347
    
    关键概念:
    - Policy Model (π_θ): 生成响应的策略网络
    - Reference Model (π_ref): 参考策略（通常是 SFT 模型），用于计算 KL 散度
    - Value Function (V): 估计状态价值
    - Advantage Function (A): 估计动作优势
    """
    
    def __init__(
        self,
        config: PPOConfig,
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
        
        # 加载策略模型（带 value head）
        logger.info(f"加载策略模型: {config.model_name}")
        self.policy_model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True
        )
        
        # 添加 value head
        self.value_head = nn.Linear(
            self.policy_model.config.hidden_size, 1
        )
        self.value_head.to(self.policy_model.device)
        
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
            * config.ppo_epochs
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
            self.train_dataloader, self.lr_scheduler, self.value_head = \
                self.accelerator.prepare(
                    self.policy_model, self.ref_model, self.optimizer,
                    self.train_dataloader, self.lr_scheduler, self.value_head
                )
        
        # 训练状态
        self.global_step = 0
        self.epoch = 0
        
        # 创建输出目录
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    
    def _apply_lora(self):
        """应用 LoRA 到策略模型"""
        if self.config.target_modules is None:
            # 自动检测目标模块
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
        # 收集需要训练的参数
        trainable_params = list(self.policy_model.parameters()) + list(self.value_head.parameters())
        trainable_params = [p for p in trainable_params if p.requires_grad]
        
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=(0.9, 0.999)
        )
        return optimizer
    
    def _collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """数据批处理"""
        # 处理 prompt_ids
        prompt_ids = [item["prompt_ids"] for item in batch]
        max_prompt_len = max(len(p) for p in prompt_ids)
        
        # Padding
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
    def generate_responses(
        self,
        prompt_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        生成响应
        
        Returns:
            response_ids: 生成的 token IDs
            response_mask: 响应的 attention mask
        """
        outputs = self.policy_model.generate(
            input_ids=prompt_ids,
            attention_mask=attention_mask,
            max_new_tokens=self.config.max_response_length,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            top_k=self.config.top_k,
            do_sample=self.config.do_sample,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True
        )
        
        response_ids = outputs.sequences[:, prompt_ids.shape[1]:]
        response_mask = (response_ids != self.tokenizer.pad_token_id).long()
        
        return response_ids, response_mask
    
    @torch.no_grad()
    def compute_rewards(
        self,
        response_ids: torch.Tensor,
        events: List[List[Dict]],
        contexts: List[str]
    ) -> torch.Tensor:
        """
        计算奖励
        """
        rewards = []
        
        for i, response in enumerate(response_ids):
            # 解码响应
            response_text = self.tokenizer.decode(
                response, skip_special_tokens=True
            )
            
            # 计算奖励
            reward_dict = self.reward_fn.compute_reward(
                response_text,
                events[i]
            )
            rewards.append(reward_dict["total"])
        
        return torch.tensor(rewards, dtype=torch.float32).to(response_ids.device)
    
    @torch.no_grad()
    def compute_logprobs_and_values(
        self,
        model,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        response_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算 log probabilities 和 values
        """
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
        
        # 计算 log probs
        logits = outputs.logits[:, :-1, :]
        log_probs = F.log_softmax(logits, dim=-1)
        
        # 获取实际 token 的 log prob
        target_ids = input_ids[:, 1:]
        token_log_probs = log_probs.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)
        
        # 只取 response 部分的 log probs
        response_mask_shifted = response_mask[:, 1:]
        token_log_probs = token_log_probs * response_mask_shifted
        
        # 计算 values (使用 value head)
        if hasattr(self, 'value_head'):
            hidden_states = outputs.hidden_states[-1]
            values = self.value_head(hidden_states).squeeze(-1)
            values = values[:, :-1] * response_mask_shifted
        else:
            values = token_log_probs.detach()
        
        return token_log_probs, values
    
    def compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        next_values: torch.Tensor,
        masks: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算 GAE (Generalized Advantage Estimation)
        
        参考文献: [1] 中的 Advantage Estimation
        """
        advantages = torch.zeros_like(rewards)
        last_gae = 0
        
        for t in reversed(range(rewards.shape[1])):
            if t == rewards.shape[1] - 1:
                next_value = next_values[:, t]
            else:
                next_value = values[:, t + 1]
            
            delta = rewards[:, t] + self.config.gamma * next_value * masks[:, t] - values[:, t]
            last_gae = delta + self.config.gamma * self.config.lam * last_gae * masks[:, t]
            advantages[:, t] = last_gae
        
        returns = advantages + values
        
        return advantages, returns
    
    def ppo_loss(
        self,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        values: torch.Tensor,
        returns: torch.Tensor,
        response_mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        计算 PPO 损失
        
        包含:
        1. Policy Loss (Clipped Surrogate Objective)
        2. Value Loss
        3. Entropy Bonus (可选)
        """
        # 1. Policy Loss
        ratio = torch.exp(log_probs - old_log_probs)
        
        # Clipped surrogate objective
        clipped_ratio = torch.clamp(
            ratio,
            1 - self.config.clip_epsilon,
            1 + self.config.clip_epsilon
        )
        
        surrogate1 = ratio * advantages
        surrogate2 = clipped_ratio * advantages
        policy_loss = -torch.min(surrogate1, surrogate2)
        policy_loss = (policy_loss * response_mask).sum() / response_mask.sum()
        
        # 2. Value Loss (Clipped)
        value_pred_clipped = old_log_probs + torch.clamp(
            values - old_log_probs,
            -self.config.value_clip,
            self.config.value_clip
        )
        value_loss1 = (values - returns) ** 2
        value_loss2 = (value_pred_clipped - returns) ** 2
        value_loss = 0.5 * torch.max(value_loss1, value_loss2)
        value_loss = (value_loss * response_mask).sum() / response_mask.sum()
        
        # 3. KL 惩罚
        kl_penalty = ((log_probs - old_log_probs) ** 2 * response_mask).sum() / response_mask.sum()
        
        total_loss = policy_loss + 0.5 * value_loss + self.config.kl_coef * kl_penalty
        
        return {
            "total_loss": total_loss,
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "kl_penalty": kl_penalty,
            "ratio": ratio.mean()
        }
    
    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """
        执行一个 PPO 训练步骤
        """
        # 1. 生成响应 (使用当前策略)
        prompt_ids = batch["prompt_ids"]
        attention_mask = batch["attention_mask"]
        
        with torch.no_grad():
            response_ids, response_mask = self.generate_responses(
                prompt_ids, attention_mask
            )
            
            # 计算奖励
            rewards = self.compute_rewards(
                response_ids,
                batch["events"],
                batch["contexts"]
            )
        
        # 2. 构建完整序列 (prompt + response)
        full_input_ids = torch.cat([prompt_ids, response_ids], dim=1)
        full_attention_mask = torch.cat([attention_mask, response_mask], dim=1)
        full_response_mask = torch.cat([
            torch.zeros_like(attention_mask),
            response_mask
        ], dim=1)
        
        # 3. 计算 old log probs (来自参考模型)
        with torch.no_grad():
            old_log_probs, old_values = self.compute_logprobs_and_values(
                self.ref_model,
                full_input_ids,
                full_attention_mask,
                full_response_mask
            )
        
        # 4. PPO 更新循环
        total_losses = []
        policy_losses = []
        value_losses = []
        kl_penalties = []
        
        for ppo_epoch in range(self.config.ppo_epochs):
            # 计算当前策略的 log probs 和 values
            log_probs, values = self.compute_logprobs_and_values(
                self.policy_model,
                full_input_ids,
                full_attention_mask,
                full_response_mask
            )
            
            # 简化: 直接使用奖励作为优势
            advantages = rewards.unsqueeze(1).expand_as(log_probs)
            returns = advantages
            
            # 计算 PPO 损失
            loss_dict = self.ppo_loss(
                log_probs,
                old_log_probs,
                advantages,
                values,
                returns,
                full_response_mask
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
            value_losses.append(loss_dict["value_loss"].item())
            kl_penalties.append(loss_dict["kl_penalty"].item())
        
        return {
            "total_loss": sum(total_losses) / len(total_losses),
            "policy_loss": sum(policy_losses) / len(policy_losses),
            "value_loss": sum(value_losses) / len(value_losses),
            "kl_penalty": sum(kl_penalties) / len(kl_penalties),
            "mean_reward": rewards.mean().item()
        }
    
    def train(self):
        """
        主训练循环
        """
        logger.info("开始 PPO 训练...")
        
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
                        f"reward={metrics['mean_reward']:.4f}"
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
    config = PPOConfig(
        num_epochs=1,
        batch_size=2,
        output_dir="outputs/ppo_test"
    )
    
    # 创建训练器并训练
    print("开始训练...")
    trainer = PPOTrainer(config, rl_data)
    trainer.train()


if __name__ == "__main__":
    main()
