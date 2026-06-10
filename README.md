# 南京大学 RL 考核项目 - PPO & GRPO 实现

本项目实现了基于 Thyme-SFT 数据集的 PPO 和 GRPO 强化学习训练流程。

## 项目结构

```
.
├── configs/                    # 配置文件目录
│   ├── ppo_config.yaml          # PPO 配置
│   └── grpo_config.yaml         # GRPO 配置
├── data/                        # 数据目录
│   ├── thyme_sft.json          # SFT 格式原始数据
│   └── thyme_rl.jsonl          # RL 格式转换后数据
├── outputs/                     # 训练输出目录
│   ├── ppo/                    # PPO 训练结果
│   └── grpo/                   # GRPO 训练结果
├── src/                         # 源代码
│   ├── __init__.py
│   ├── data_download.py         # 数据下载/生成
│   ├── data_converter.py          # SFT 到 RL 格式转换
│   ├── ppo_trainer.py           # PPO 训练器
│   └── grpo_trainer.py          # GRPO 训练器
├── train_ppo.py                 # PPO 训练脚本
├── train_grpo.py                # GRPO 训练脚本
├── requirements.txt             # 依赖包
└── README.md                    # 项目说明
```

## 环境配置

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 模型下载

项目使用 `Qwen/Qwen2.5-VL-3B-Instruct` 作为基础模型，首次运行时会自动从 Hugging Face 下载。

**国内用户配置镜像（可选）：**

```bash
# Windows PowerShell
$env:HF_ENDPOINT = "https://hf-mirror.com"

# Linux/Mac
export HF_ENDPOINT=https://hf-mirror.com
```

## 快速开始

### 1. 数据准备

运行任意训练脚本时会自动下载/生成数据：

```bash
huggingface-cli download Kwai-Keye/Thyme-SFT data/2round-00000-of-00069.parquet --local-dir ./data 
```

### 2.数据可视化
```bash
python src/transfer/viewer.py
```

### 3.数据转换
从 SFT 轨迹格式数据转换到RL问答格式数据

```bash
 python src/transfer/data_converter.py --limit 1000 --no-tokenizer
```
### 4. PPO 训练

```bash
# 使用默认配置
python train_ppo.py

# 使用自定义配置
python train_ppo.py --config configs/ppo_config.yaml

# 命令行参数覆盖
python train_ppo.py \
    --model_name Qwen/Qwen2.5-VL-3B-Instruct \
    --num_epochs 5 \
    --batch_size 4 \
    --learning_rate 5e-5 \
    --output_dir outputs/ppo_v1
```

### 5. GRPO 训练

```bash
# 使用默认配置
python train_grpo.py

# 启用 Online Filter（Bonus 题目 5a）
python train_grpo.py --use_online_filter --adv_var_low 0.01 --adv_var_high 1.0

# 自定义组大小 G
python train_grpo.py --group_size 8 --batch_size 2
```

### 4. 对比实验

```bash
# 同时运行两种算法
# 终端 1: PPO
python train_ppo.py --output_dir outputs/ppo_compare

# 终端 2: GRPO
python train_grpo.py --output_dir outputs/grpo_compare
```

## 算法说明

### PPO (Proximal Policy Optimization)

PPO 是一种 on-policy 强化学习算法，主要特点：

- **Value Function**: 需要单独训练价值函数估计状态价值
- **Advantage Estimation**: 使用 GAE (Generalized Advantage Estimation)
- **Policy Update**: 使用 clipped surrogate objective 限制策略更新幅度
- **KL Penalty**: 通过 KL 散度惩罚防止策略偏离参考模型太远

核心公式：
```
L^CLIP(θ) = E[min(ratio * A, clip(ratio, 1-ε, 1+ε) * A)]
```

### GRPO (Group Relative Policy Optimization)

GRPO 是 DeepSeekMath 论文中提出的算法，主要改进：

- **无需 Value Function**: 通过组采样避免训练单独的价值函数
- **Group Sampling**: 对每个问题采样 G 个回答，计算组内相对优势
- **Relative Advantage**: 使用标准化组内奖励作为优势估计
- **更稳定的训练**: 通常比 PPO 更稳定，学习率可以更小

核心公式：
```
A_i = (r_i - mean(r)) / std(r)   # 组内相对优势
L^GRPO(θ) = E[min(ratio * A, clip(ratio, 1-ε, 1+ε) * A)] - β * KL(π_θ || π_ref)
```

## 配置参数说明

### PPO 特有参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `ppo_epochs` | 每次采样的更新轮数 | 4 |
| `value_clip` | Value function clipping | 0.2 |
| `gamma` | 折扣因子 | 1.0 |
| `lam` | GAE lambda | 0.95 |

### GRPO 特有参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `group_size` | 每个问题的采样数 G | 4 |
| `grpo_epochs` | 每次采样的更新轮数 | 2 |
| `kl_coef` | KL 惩罚系数（通常比 PPO 小） | 0.04 |

### Online Filter 参数 (Bonus)

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `use_online_filter` | 启用优势方差过滤 | false |
| `advantage_var_low` | 优势方差下限 | 0.01 |
| `advantage_var_high` | 优势方差上限 | 1.0 |

## 奖励函数设计

针对医学时间线提取任务，奖励函数包含三个维度：

1. **格式奖励 (20%)**: 检查输出是否为有效 JSON 格式
2. **事件覆盖奖励 (40%)**: 评估提取的事件完整性和准确性
3. **时间准确性奖励 (40%)**: 评估时间信息匹配程度

## 实验结果对比

### PPO 优势与劣势

**优势:**
- 理论基础扎实，有完整的状态价值估计
- GAE 提供更准确的优势估计
- 在复杂任务上表现稳定

**劣势:**
- 需要训练单独的价值函数
- 计算成本更高（需要额外的 value head）
- 超参数调整更复杂

### GRPO 优势与劣势

**优势:**
- 无需价值函数，简化实现
- 组采样提供自然的基线
- 训练更稳定，对超参数不那么敏感
- 内存和计算效率更高

**劣势:**
- 依赖组大小 G，G 太小会导致估计方差大
- 对某些任务可能需要更多的采样

## 参考论文

1. Schulman et al., "Proximal Policy Optimization Algorithms", arXiv:1707.06347, 2017.
2. Schulman et al., "Trust Region Policy Optimization", ICML 2015.
3. Shao et al., "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models", arXiv:2402.03300, 2024.
4. Dong et al., "Agentic Reinforced Policy Optimization", arXiv:2507.19849, 2025.

## 注意事项

1. **显存要求**: 使用 Qwen-2.5-VL-3B-Instruct 需要至少 12GB 显存，使用 LoRA 可降低要求
2. **训练时间**: 1000 条数据训练 3 个 epoch 大约需要 2-4 小时（取决于硬件）
3. **数据格式**: 转换后的 RL 格式包含 `prompt_ids`，节省 tokenization 时间

## 问题排查

### 显存不足

```bash
# 减小 batch_size 和序列长度
python train_ppo.py --batch_size 2 --max_seq_length 1024
```

### 模型下载失败

```bash
# 配置 Hugging Face 镜像
export HF_ENDPOINT=https://hf-mirror.com
python train_ppo.py
```

### 奖励不提升

- 检查奖励函数实现是否正确
- 调整 KL 系数（太大限制探索，太小容易偏离）
- 调整学习率

## License

本项目仅供学术研究和学习使用。
