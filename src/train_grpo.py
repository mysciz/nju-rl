"""
GRPO 训练主脚本
使用示例: python train_grpo.py --config configs/grpo_config.yaml
"""
import os
import sys
import json
import argparse
import yaml
from pathlib import Path

# 添加 src 到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data_download import generate_timeline_data
from data_converter import SFTtoRLConverter, RLDataItem
from grpo_trainer import GRPOTrainer, GRPOConfig


def load_config(config_path: str) -> dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        if config_path.endswith('.yaml') or config_path.endswith('.yml'):
            return yaml.safe_load(f)
        else:
            return json.load(f)


def prepare_data(num_samples: int = 1000, data_dir: str = "data"):
    """准备训练数据"""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    sft_path = data_dir / "thyme_sft.json"
    rl_path = data_dir / "thyme_rl.jsonl"
    
    if sft_path.exists():
        print(f"加载已有 SFT 数据: {sft_path}")
        with open(sft_path, 'r', encoding='utf-8') as f:
            sft_data = json.load(f)
    else:
        print(f"生成 {num_samples} 条 SFT 数据...")
        sft_data = generate_timeline_data(num_samples=num_samples)
        
        with open(sft_path, 'w', encoding='utf-8') as f:
            json.dump(sft_data, f, ensure_ascii=False, indent=2)
        print(f"SFT 数据已保存到: {sft_path}")
    
    if rl_path.exists():
        print(f"加载已有 RL 数据: {rl_path}")
        rl_data = []
        with open(rl_path, 'r', encoding='utf-8') as f:
            for line in f:
                rl_data.append(json.loads(line))
    else:
        print("转换 SFT 数据到 RL 格式...")
        converter = SFTtoRLConverter()
        rl_data = converter.convert_sft_to_rl(sft_data, output_path=str(rl_path))
        
        rl_data = [item.to_dict() if hasattr(item, 'to_dict') else item for item in rl_data]
        print(f"RL 数据已保存到: {rl_path}")
    
    return sft_data, rl_data


def main():
    parser = argparse.ArgumentParser(description="GRPO 训练脚本")
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="configs/grpo_config.yaml",
        help="配置文件路径"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-VL-3B-Instruct",
        help="基础模型名称"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/grpo",
        help="输出目录"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1000,
        help="数据样本数量"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=3,
        help="训练轮数"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="批次大小（问题数）"
    )
    parser.add_argument(
        "--group_size",
        type=int,
        default=4,
        help="每个问题的采样数（G）"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="学习率"
    )
    parser.add_argument(
        "--lora_r",
        type=int,
        default=16,
        help="LoRA rank"
    )
    parser.add_argument(
        "--kl_coef",
        type=float,
        default=0.04,
        help="KL 散度系数"
    )
    parser.add_argument(
        "--use_online_filter",
        action="store_true",
        help="启用 online filter（Bonus 题目）"
    )
    parser.add_argument(
        "--adv_var_low",
        type=float,
        default=0.01,
        help="优势方差下限"
    )
    parser.add_argument(
        "--adv_var_high",
        type=float,
        default=1.0,
        help="优势方差上限"
    )
    
    args = parser.parse_args()
    
    # 加载配置
    if os.path.exists(args.config):
        print(f"加载配置文件: {args.config}")
        config_dict = load_config(args.config)
    else:
        print(f"配置文件不存在，使用默认配置: {args.config}")
        config_dict = {}
    
    # 使用命令行参数覆盖配置
    config_dict.update({
        "model_name": args.model_name,
        "output_dir": args.output_dir,
        "num_epochs": args.num_epochs,
        "batch_size": args.batch_size,
        "group_size": args.group_size,
        "learning_rate": args.learning_rate,
        "lora_r": args.lora_r,
        "kl_coef": args.kl_coef,
        "use_online_filter": args.use_online_filter,
        "advantage_var_low": args.adv_var_low,
        "advantage_var_high": args.adv_var_high
    })
    
    # 创建 GRPOConfig
    config = GRPOConfig(**config_dict)
    
    print("=" * 50)
    print("GRPO 训练配置:")
    print(f"  模型: {config.model_name}")
    print(f"  输出目录: {config.output_dir}")
    print(f"  训练轮数: {config.num_epochs}")
    print(f"  批次大小（问题数）: {config.batch_size}")
    print(f"  组大小 G: {config.group_size}")
    print(f"  学习率: {config.learning_rate}")
    print(f"  LoRA r: {config.lora_r}")
    print(f"  KL 系数: {config.kl_coef}")
    print(f"  Online Filter: {config.use_online_filter}")
    if config.use_online_filter:
        print(f"    方差范围: [{config.advantage_var_low}, {config.advantage_var_high}]")
    print("=" * 50)
    
    # 准备数据
    print("\n准备训练数据...")
    _, rl_data = prepare_data(num_samples=args.num_samples)
    
    # 将 dict 转换回 RLDataItem
    rl_data_items = []
    for item in rl_data:
        rl_data_items.append(RLDataItem(
            id=item["id"],
            query=item["query"],
            context=item["context"],
            reference_answer=item["reference_answer"],
            system_prompt=item["system_prompt"],
            prompt=item["prompt"],
            prompt_ids=item["prompt_ids"],
            reference_ids=item.get("reference_ids"),
            events=item["events"]
        ))
    
    # 创建训练器
    print("\n初始化 GRPO 训练器...")
    trainer = GRPOTrainer(config, rl_data_items)
    
    # 开始训练
    print("\n开始训练...")
    trainer.train()
    
    print("\n训练完成!")


if __name__ == "__main__":
    main()
