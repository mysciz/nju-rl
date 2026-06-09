"""
Thyme-SFT 数据集查看器和可视化工具
用于加载、统计和可视化 SFT 格式数据
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import Counter
import re


class SFTDataViewer:
    """
    Thyme-SFT 数据集查看器
    
    支持:
    - 加载 parquet 格式的 SFT 数据
    - 统计数据集基本信息
    - 可视化数据分布
    - 展示样本示例
    """
    
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        self.df = None
        self.stats = {}
        
    def load_data(self) -> pd.DataFrame:
        """加载 parquet 数据"""
        print(f"加载数据: {self.data_path}")
        self.df = pd.read_parquet(self.data_path)
        print(f"成功加载 {len(self.df)} 条数据")
        return self.df
    
    def compute_stats(self) -> Dict[str, Any]:
        """计算数据集统计信息"""
        if self.df is None:
            self.load_data()
        
        stats = {
            "总样本数": len(self.df),
            "列名": list(self.df.columns),
            "数据类型": {col: str(self.df[col].dtype) for col in self.df.columns},
        }
        
        # 文本长度统计
        if "question" in self.df.columns:
            question_lengths = self.df["question"].astype(str).apply(len)
            stats["question 长度统计"] = {
                "平均长度": round(question_lengths.mean(), 2),
                "中位数长度": int(question_lengths.median()),
                "最短": int(question_lengths.min()),
                "最长": int(question_lengths.max()),
            }
        
        if "response" in self.df.columns:
            response_lengths = self.df["response"].astype(str).apply(len)
            stats["response 长度统计"] = {
                "平均长度": round(response_lengths.mean(), 2),
                "中位数长度": int(response_lengths.median()),
                "最短": int(response_lengths.min()),
                "最长": int(response_lengths.max()),
            }
        
        # 图像统计
        if "image" in self.df.columns:
            has_image = self.df["image"].apply(lambda x: x is not None and len(str(x)) > 0)
            stats["图像统计"] = {
                "有图像的样本数": int(has_image.sum()),
                "无图像的样本数": int((~has_image).sum()),
            }
        
        self.stats = stats
        return stats
    
    def print_stats(self):
        """打印统计数据"""
        if not self.stats:
            self.compute_stats()
        
        print("\n" + "="*80)
        print("数据集统计信息")
        print("="*80)
        
        for key, value in self.stats.items():
            if isinstance(value, dict):
                print(f"\n【{key}】")
                for k, v in value.items():
                    print(f"  {k}: {v}")
            else:
                print(f"{key}: {value}")
        
        print("="*80)
    
    def show_sample(self, index: int = 0, verbose: bool = True) -> Dict[str, Any]:
        """
        展示单个样本
        
        Args:
            index: 样本索引
            verbose: 是否打印详细信息
        """
        if self.df is None:
            self.load_data()
        
        if index >= len(self.df):
            print(f"索引 {index} 超出范围，数据集共有 {len(self.df)} 条")
            return {}
        
        sample = self.df.iloc[index].to_dict()
        
        if verbose:
            print(f"\n{'='*80}")
            print(f"样本 {index} 详情")
            print(f"{'='*80}")
            
            for col, value in sample.items():
                if col == "image":
                    # 图像数据太长，只显示前100字符
                    has_value = value is not None
                    if isinstance(value, list):
                        has_value = len(value) > 0
                    elif hasattr(value, '__len__') and not isinstance(value, (str, bytes)):
                        has_value = len(value) > 0
                    
                    img_preview = str(value)[:200] if has_value else "None"
                    print(f"\n【{col}】")
                    print(f"  类型: {type(value)}")
                    if isinstance(value, list):
                        print(f"  图像数量: {len(value)}")
                        if len(value) > 0:
                            print(f"  第一张图像预览: {str(value[0])[:150]}...")
                    else:
                        print(f"  预览: {img_preview}...")
                elif col in ["question", "response"]:
                    print(f"\n【{col}】")
                    print(f"  长度: {len(str(value))} 字符")
                    print(f"  内容:")
                    # 格式化显示文本，每行最多80字符
                    text = str(value)
                    if len(text) > 500:
                        text = text[:500] + "\n  ... (内容已截断)"
                    for line in text.split('\n'):
                        print(f"    {line}")
                else:
                    print(f"\n【{col}】: {value}")
            
            print(f"{'='*80}\n")
        
        return sample
    
    def show_multiple_samples(self, indices: Optional[List[int]] = None, num: int = 3):
        """
        展示多个样本
        
        Args:
            indices: 指定索引列表，为 None 则随机选择
            num: 随机选择的样本数
        """
        if self.df is None:
            self.load_data()
        
        if indices is None:
            import random
            indices = random.sample(range(len(self.df)), min(num, len(self.df)))
        
        print(f"\n展示 {len(indices)} 个样本:\n")
        for idx in indices:
            self.show_sample(idx, verbose=True)
    
    def analyze_question_types(self) -> Dict[str, int]:
        """分析问题类型分布"""
        if self.df is None:
            self.load_data()
        
        if "question" not in self.df.columns:
            return {}
        
        # Extract question keywords
        questions = self.df["question"].astype(str)
        
        # Define keywords for question type classification
        keywords = {
            "Time/Date": ["when", "time", "date", "day", "month", "year"],
            "Diagnosis": ["diagnosis", "diagnosed"],
            "Medication": ["medication", "drug", "medicine", "dose"],
            "Procedure": ["procedure", "surgery", "operation"],
            "Symptom": ["symptom", "pain", "fever"],
            "Admission": ["admission", "discharge", "admitted"],
            "Location": ["where", "location", "side", "left", "right"],
            "Yes/No": ["is there", "does", "has", "have", "can", "is the", "are the"],
        }
        
        type_counts = Counter()
        
        for q in questions:
            q_lower = q.lower()
            matched = []
            for q_type, words in keywords.items():
                if any(word in q_lower for word in words):
                    matched.append(q_type)
            
            if matched:
                type_counts["+".join(sorted(matched))] += 1
            else:
                type_counts["Others"] += 1
        
        return dict(type_counts)
    
    def visualize(self, save_path: Optional[str] = None):
        """
        可视化数据分布
        
        Args:
            save_path: 图表保存路径，为 None 则显示图表
        """
        if self.df is None:
            self.load_data()
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Thyme-SFT Dataset Visualization", fontsize=16)
        
        # 1. Text Length Distribution - Question
        ax1 = axes[0, 0]
        if "question" in self.df.columns:
            question_lengths = self.df["question"].astype(str).apply(len)
            ax1.hist(question_lengths, bins=30, alpha=0.7, color='blue', label='Question')
            ax1.set_xlabel("Characters")
            ax1.set_ylabel("Frequency")
            ax1.set_title("Question Length Distribution")
            ax1.axvline(question_lengths.mean(), color='red', linestyle='--', label=f'Mean: {question_lengths.mean():.0f}')
            ax1.legend()
        
        # 2. Text Length Distribution - Response
        ax2 = axes[0, 1]
        if "response" in self.df.columns:
            response_lengths = self.df["response"].astype(str).apply(len)
            ax2.hist(response_lengths, bins=30, alpha=0.7, color='green', label='Response')
            ax2.set_xlabel("Characters")
            ax2.set_ylabel("Frequency")
            ax2.set_title("Response Length Distribution")
            ax2.axvline(response_lengths.mean(), color='red', linestyle='--', label=f'Mean: {response_lengths.mean():.0f}')
            ax2.legend()
        
        # 3. Question Type Distribution
        ax3 = axes[1, 0]
        q_types = self.analyze_question_types()
        if q_types:
            # Translate Chinese keys to English
            type_mapping = {
                "时间/日期": "Time/Date",
                "诊断": "Diagnosis",
                "用药": "Medication",
                "手术/操作": "Procedure",
                "症状": "Symptom",
                "入院/出院": "Admission/Discharge",
                "位置": "Location",
                "判断/是或否": "Yes/No",
                "其他": "Others"
            }
            translated_types = []
            for t in list(q_types.keys())[:8]:
                # Split by + and translate each part
                parts = t.split("+")
                translated = "+".join([type_mapping.get(p, p) for p in parts])
                translated_types.append(translated)
            counts = [q_types[t] for t in list(q_types.keys())[:8]]
            ax3.barh(translated_types, counts, color='orange')
            ax3.set_xlabel("Count")
            ax3.set_title("Question Type Distribution (Top 8)")
            ax3.invert_yaxis()
        
        # 4. Data Statistics Overview
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        # Prepare stats text in English
        if not self.stats:
            self.compute_stats()
        
        stats_text = "Dataset Overview\n" + "="*30 + "\n\n"
        stats_text += f"Total Samples: {self.stats.get('总样本数', 'N/A')}\n\n"
        
        if "question 长度统计" in self.stats:
            q_stat = self.stats["question 长度统计"]
            stats_text += "Question Length:\n"
            stats_text += f"  Mean: {q_stat['平均长度']:.0f}\n"
            stats_text += f"  Range: {q_stat['最短']} - {q_stat['最长']}\n\n"
        
        if "response 长度统计" in self.stats:
            r_stat = self.stats["response 长度统计"]
            stats_text += "Response Length:\n"
            stats_text += f"  Mean: {r_stat['平均长度']:.0f}\n"
            stats_text += f"  Range: {r_stat['最短']} - {r_stat['最长']}\n\n"
        
        if "图像统计" in self.stats:
            img_stat = self.stats["图像统计"]
            stats_text += "Image Statistics:\n"
            stats_text += f"  With Images: {img_stat['有图像的样本数']}\n"
            stats_text += f"  Without Images: {img_stat['无图像的样本数']}\n"
        
        ax4.text(0.1, 0.5, stats_text, fontsize=11, verticalalignment='center',
                family='monospace', transform=ax4.transAxes)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"图表已保存到: {save_path}")
        else:
            plt.show()
        
        return fig
    
    def save_sample_json(self, output_path: str, num_samples: int = 5):
        """
        保存样本到 JSON 文件
        
        Args:
            output_path: 输出文件路径
            num_samples: 保存的样本数
        """
        if self.df is None:
            self.load_data()
        
        samples = []
        for i in range(min(num_samples, len(self.df))):
            sample = self.df.iloc[i].to_dict()
            # 将图像数据转换为标记，避免 JSON 太大
            if "image" in sample and sample["image"] is not None:
                if isinstance(sample["image"], list):
                    sample["image"] = f"<图像列表，共 {len(sample['image'])} 张>"
                else:
                    sample["image"] = f"<图像数据，长度 {len(str(sample['image']))}>"
            samples.append(sample)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(samples, f, ensure_ascii=False, indent=2)
        
        print(f"已保存 {num_samples} 个样本到: {output_path}")


def main():
    """
    主函数：加载并可视化 Thyme-SFT 数据集
    """
    # 数据路径
    data_path = "/root/nju-rl/data/2round-00000-of-00069.parquet"
    
    # 创建查看器
    viewer = SFTDataViewer(data_path)
    
    # 加载并统计数据
    print("="*80)
    print("Thyme-SFT 数据集分析")
    print("="*80)
    
    viewer.load_data()
    viewer.compute_stats()
    viewer.print_stats()
    
    # 显示样本
    print("\n显示 3 个随机样本:\n")
    viewer.show_multiple_samples(num=3)
    
    # 保存样本到 JSON 方便查看
    output_dir = Path("/root/nju-rl/data")
    output_dir.mkdir(exist_ok=True)
    viewer.save_sample_json(output_dir / "sample_visualization.json", num_samples=5)
    
    # 生成可视化图表
    print("\n生成可视化图表...")
    try:
        viewer.visualize(save_path=str(output_dir / "dataset_visualization.png"))
    except Exception as e:
        print(f"可视化图表生成失败: {e}")
        print("这可能是因为没有 GUI 环境，但数据分析已完成")
    
    print("\n" + "="*80)
    print("分析完成!")
    print("="*80)


if __name__ == "__main__":
    main()
