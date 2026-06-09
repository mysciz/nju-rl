"""
NJU RL 考核项目 - PPO & GRPO 实现

本项目实现了：
1. Thyme-SFT 数据下载和生成
2. SFT 到 RL 格式的转换
3. PPO 训练器
4. GRPO 训练器

作者: RL Lab Assignment
"""

__version__ = "1.0.0"

from .data_download import download_thyme_sft, generate_timeline_data
from .data_converter import SFTtoRLConverter, RewardFunction, RLDataItem
from .ppo_trainer import PPOTrainer, PPOConfig
from .grpo_trainer import GRPOTrainer, GRPOConfig

__all__ = [
    "download_thyme_sft",
    "generate_timeline_data",
    "SFTtoRLConverter",
    "RewardFunction",
    "RLDataItem",
    "PPOTrainer",
    "PPOConfig",
    "GRPOTrainer",
    "GRPOConfig",
]
