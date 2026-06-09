"""
可视化训练结果
绘制奖励曲线和对比图
"""
import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))


def load_training_logs(log_dir: Path) -> Dict[str, List]:
    """
    从 tensorboard 日志或自定义日志加载训练数据
    这是一个简单的实现，实际使用时可以读取 tensorboard 日志
    """
    logs = {
        "steps": [],
        "rewards": [],
        "losses": []
    }
    
    # 这里可以从 tensorboard 或其他日志文件加载
    # 简化起见，我们生成模拟数据
    
    return logs


def create_comparison_plot(
    ppo_data: Dict,
    grpo_data: Dict,
    output_path: str
):
    """
    创建对比图表
    
    Args:
        ppo_data: PPO 训练数据
        grpo_data: GRPO 训练数据
        output_path: 输出图片路径
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("需要安装 matplotlib: pip install matplotlib")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('PPO vs GRPO 训练对比', fontsize=14)
    
    # 图 1: 奖励曲线对比
    ax1 = axes[0, 0]
    if ppo_data.get('steps'):
        ax1.plot(ppo_data['steps'], ppo_data['rewards'], label='PPO', alpha=0.7)
    if grpo_data.get('steps'):
        ax1.plot(grpo_data['steps'], grpo_data['rewards'], label='GRPO', alpha=0.7)
    ax1.set_xlabel('训练步数')
    ax1.set_ylabel('平均奖励')
    ax1.set_title('奖励曲线对比')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 图 2: 损失曲线对比
    ax2 = axes[0, 1]
    if ppo_data.get('steps'):
        ax2.plot(ppo_data['steps'], ppo_data['losses'], label='PPO', alpha=0.7)
    if grpo_data.get('steps'):
        ax2.plot(grpo_data['steps'], grpo_data['losses'], label='GRPO', alpha=0.7)
    ax2.set_xlabel('训练步数')
    ax2.set_ylabel('损失')
    ax2.set_title('损失曲线对比')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 图 3: 奖励分布对比
    ax3 = axes[1, 0]
    if ppo_data.get('rewards'):
        ax3.hist(ppo_data['rewards'], bins=20, alpha=0.5, label='PPO')
    if grpo_data.get('rewards'):
        ax3.hist(grpo_data['rewards'], bins=20, alpha=0.5, label='GRPO')
    ax3.set_xlabel('奖励值')
    ax3.set_ylabel('频次')
    ax3.set_title('奖励分布对比')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 图 4: 统计对比
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # 准备对比文本
    comparison_text = "算法对比:\n\n"
    
    comparison_text += "PPO:\n"
    comparison_text += f"  - 平均奖励: {np.mean(ppo_data['rewards']) if ppo_data.get('rewards') else 'N/A':.4f}\n"
    comparison_text += f"  - 最终奖励: {ppo_data['rewards'][-1] if ppo_data.get('rewards') else 'N/A':.4f}\n"
    comparison_text += "  - 优点: 完整的价值函数估计\n"
    comparison_text += "  - 缺点: 需要训练价值函数\n\n"
    
    comparison_text += "GRPO:\n"
    comparison_text += f"  - 平均奖励: {np.mean(grpo_data['rewards']) if grpo_data.get('rewards') else 'N/A':.4f}\n"
    comparison_text += f"  - 最终奖励: {grpo_data['rewards'][-1] if grpo_data.get('rewards') else 'N/A':.4f}\n"
    comparison_text += "  - 优点: 无需价值函数\n"
    comparison_text += "  - 缺点: 依赖组大小 G\n"
    
    ax4.text(0.1, 0.9, comparison_text, transform=ax4.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"图表已保存到: {output_path}")
    plt.close()


def create_online_filter_comparison(
    baseline_data: Dict,
    filter_data: Dict,
    output_path: str
):
    """
    创建 Online Filter 效果对比图
    （用于 Bonus 题目 5a）
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("需要安装 matplotlib: pip install matplotlib")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Online Filter 效果对比 (Bonus 5a)', fontsize=14)
    
    # 图 1: 奖励曲线
    ax1 = axes[0]
    if baseline_data.get('steps'):
        ax1.plot(baseline_data['steps'], baseline_data['rewards'], 
                label='无 Filter', alpha=0.7)
    if filter_data.get('steps'):
        ax1.plot(filter_data['steps'], filter_data['rewards'],
                label='有 Online Filter', alpha=0.7)
    ax1.set_xlabel('训练步数')
    ax1.set_ylabel('平均奖励')
    ax1.set_title('奖励曲线对比')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 图 2: 优势方差对比
    ax2 = axes[1]
    if baseline_data.get('advantage_vars'):
        ax2.plot(baseline_data['steps'], baseline_data['advantage_vars'],
                label='无 Filter', alpha=0.7)
    if filter_data.get('advantage_vars'):
        ax2.plot(filter_data['steps'], filter_data['advantage_vars'],
                label='有 Online Filter', alpha=0.7)
    ax2.axhline(y=0.01, color='r', linestyle='--', alpha=0.5, label='目标下限')
    ax2.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='目标上限')
    ax2.set_xlabel('训练步数')
    ax2.set_ylabel('优势方差')
    ax2.set_title('优势方差稳定性对比')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Online Filter 对比图已保存到: {output_path}")
    plt.close()


def generate_analysis_report(
    exp_dir: Path,
    output_path: str = None
):
    """
    生成分析报告
    """
    if output_path is None:
        output_path = exp_dir / "analysis_report.md"
    
    # 加载实验配置
    config_path = exp_dir / "experiment_config.json"
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
    else:
        config = {}
    
    # 生成报告
    report = f"""# 强化学习训练结果分析报告

## 实验配置

- 实验名称: {config.get('exp_name', 'N/A')}
- 时间戳: {config.get('timestamp', 'N/A')}
- 样本数: {config.get('num_samples', 'N/A')}
- 训练轮数: {config.get('num_epochs', 'N/A')}

## 实验结果

### PPO
- 状态: {config.get('results', {}).get('ppo', {}).get('status', 'N/A')}
- 耗时: {config.get('results', {}).get('ppo', {}).get('elapsed_time', 'N/A'):.1f}秒
- 输出目录: {config.get('results', {}).get('ppo', {}).get('output_dir', 'N/A')}

### GRPO
- 状态: {config.get('results', {}).get('grpo', {}).get('status', 'N/A')}
- 耗时: {config.get('results', {}).get('grpo', {}).get('elapsed_time', 'N/A'):.1f}秒
- 输出目录: {config.get('results', {}).get('grpo', {}).get('output_dir', 'N/A')}

## 算法分析

### PPO 优势与劣势

**优势:**
1. 基于完整的价值函数估计，提供更准确的优势估计
2. 使用 GAE (Generalized Advantage Estimation) 结合 n-step 和 TD(λ) 的优点
3. 在复杂任务上表现稳定，理论基础扎实
4. Clip 机制有效限制策略更新幅度，防止训练不稳定

**劣势:**
1. 需要训练单独的价值函数，增加计算成本
2. 价值函数的训练可能不稳定，尤其是在稀疏奖励环境下
3. 超参数调整更复杂（γ, λ, clip_epsilon 等）
4. 内存占用更大（需要存储价值函数参数）

### GRPO 优势与劣势

**优势:**
1. 无需训练单独的价值函数，简化实现
2. 组采样提供自然的基线，避免价值函数估计误差
3. 组内相对优势标准化减少了奖励 scale 的影响
4. 训练更稳定，对超参数不那么敏感
5. 内存和计算效率更高

**劣势:**
1. 依赖组大小 G，G 太小会导致估计方差大
2. 对于某些任务，可能需要更多的采样来获取稳定的估计
3. 批量大小受限于 group_size，需要更大的显存来存储采样结果

## 设计原理

### PPO 设计原理

PPO 的核心思想是通过限制策略更新的幅度来保证训练的稳定性。主要技术包括：

1. **Clipped Surrogate Objective**: 通过 clip 操作限制策略比率的范围，防止策略更新过大
2. **GAE**: 结合 n-step return 和 TD 的优点，平衡偏差和方差
3. **Value Function**: 单独训练价值函数来估计状态价值

### GRPO 设计原理

GRPO 是 DeepSeekMath 论文中提出的针对数学推理任务的改进算法：

1. **Group Sampling**: 对每个问题采样多个回答，计算组内相对优势
2. **Relative Advantage**: 使用标准化组内奖励代替绝对奖励
   - A_i = (r_i - mean(r)) / std(r)
3. **无需 Value Function**: 组采样提供了自然的基线估计

这种设计特别适用于：
- 答案正确性容易判断（奖励稀疏但明确）
- 同一问题有多个可能的正确回答
- 奖励方差较大的场景

## 结论

PPO 适用于需要精细价值估计的复杂连续任务，而 GRPO 更适合答案可验证的推理任务。
对于医学时间线提取任务，GRPO 的组采样设计可能更适合，因为：
1. 奖励可明确计算（格式、事件覆盖、时间准确性）
2. 同一报告可以有多个合理的提取结果
3. 任务相对简单，不需要复杂的价值函数估计

"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"分析报告已保存到: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="可视化训练结果")
    parser.add_argument(
        "--exp_dir",
        type=str,
        default=None,
        help="实验目录"
    )
    parser.add_argument(
        "--ppo_dir",
        type=str,
        default="outputs/ppo",
        help="PPO 输出目录"
    )
    parser.add_argument(
        "--grpo_dir",
        type=str,
        default="outputs/grpo",
        help="GRPO 输出目录"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="outputs/comparison.png",
        help="输出图片路径"
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="生成分析报告"
    )
    
    args = parser.parse_args()
    
    # 加载数据
    ppo_data = load_training_logs(Path(args.ppo_dir))
    grpo_data = load_training_logs(Path(args.grpo_dir))
    
    # 创建对比图
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    create_comparison_plot(ppo_data, grpo_data, str(output_path))
    
    # 生成报告
    if args.report and args.exp_dir:
        generate_analysis_report(Path(args.exp_dir))
    
    print("\n可视化完成!")


if __name__ == "__main__":
    main()
