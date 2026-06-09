"""
Thyme-SFT 数据集下载器
该数据集用于时间线提取和医学时间推理任务
"""
import os
import json
import random
from pathlib import Path
from typing import List, Dict, Any
from datasets import load_dataset
import urllib.request


def download_thyme_sft(output_dir: str = "data", num_samples: int = 1000):
    """
    下载 Thyme-SFT 数据集（或类似的时间线推理数据集）
    由于 Thyme-SFT 可能不是公开的，这里使用类似的时间线推理数据集
    作为替代，使用以下方式构建数据：
    1. 从 THYME corpus 相关资源构建
    2. 使用时间线推理任务的数据格式
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 尝试加载时间线推理相关数据集
    try:
        # 尝试从 Hugging Face 加载时间线数据集
        dataset = load_dataset("bio-nlp/thyme", split="train", trust_remote_code=True)
    except:
        # 如果无法加载，创建模拟的时间线推理数据
        print("使用模拟的时间线推理数据...")
        dataset = generate_timeline_data(num_samples)
    
    # 保存原始数据
    raw_data_path = output_path / "thyme_sft_raw.json"
    if hasattr(dataset, 'to_json'):
        dataset.to_json(raw_data_path)
    else:
        with open(raw_data_path, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)
    
    print(f"原始数据已保存到: {raw_data_path}")
    return dataset


def generate_timeline_data(num_samples: int = 1000) -> List[Dict[str, Any]]:
    """
    生成模拟的时间线推理数据
    基于医学报告中的时间信息提取任务
    """
    # 医学时间线模板
    templates = [
        {
            "context": "患者于 {date1} 入院，主诉 {symptom1}。{date2} 进行 {procedure} 检查。",
            "events": [
                {"event": "入院", "date": "{date1}", "type": "admission"},
                {"event": "{symptom1}", "date": "{date1}", "type": "symptom"},
                {"event": "{procedure}", "date": "{date2}", "type": "procedure"}
            ]
        },
        {
            "context": "{date1} 患者开始出现 {symptom1}，随后于 {date2} 加重。",
            "events": [
                {"event": "{symptom1}出现", "date": "{date1}", "type": "symptom_onset"},
                {"event": "{symptom1}加重", "date": "{date2}", "type": "symptom_worsening"}
            ]
        },
        {
            "context": "患者既往史：{date1} 诊断为 {disease1}，{date2} 开始服用 {medication1}。",
            "events": [
                {"event": "诊断{disease1}", "date": "{date1}", "type": "diagnosis"},
                {"event": "开始服药{medication1}", "date": "{date2}", "type": "medication_start"}
            ]
        }
    ]
    
    # 医学术语
    symptoms = ["胸痛", "呼吸困难", "头痛", "发热", "恶心", "腹痛", "乏力"]
    procedures = ["CT扫描", "MRI", "血液检查", "心电图", "超声检查", "X光"]
    diseases = ["高血压", "糖尿病", "冠心病", "肺炎", "胃溃疡"]
    medications = ["阿司匹林", "二甲双胍", "降压药", "抗生素"]
    
    # 日期生成器
    def random_date():
        year = random.randint(2019, 2024)
        month = random.randint(1, 12)
        day = random.randint(1, 28)
        return f"{year}年{month}月{day}日"
    
    data = []
    for i in range(num_samples):
        template = random.choice(templates)
        
        # 填充模板
        date1 = random_date()
        date2 = random_date()
        symptom1 = random.choice(symptoms)
        procedure = random.choice(procedures)
        disease1 = random.choice(diseases)
        medication1 = random.choice(medications)
        
        context = template["context"].format(
            date1=date1, date2=date2, symptom1=symptom1,
            procedure=procedure, disease1=disease1, medication1=medication1
        )
        
        # 格式化事件
        events = []
        for event in template["events"]:
            event_text = event["event"].format(
                symptom1=symptom1, procedure=procedure,
                disease1=disease1, medication1=medication1
            )
            events.append({
                "event": event_text,
                "date": event["date"].format(date1=date1, date2=date2),
                "type": event["type"]
            })
        
        # SFT 格式的轨迹数据
        sft_item = {
            "id": f"thyme_{i:04d}",
            "context": context,
            "instruction": "从上述医学报告中提取时间线信息，识别关键事件及其发生时间。",
            "input": context,
            "output": json.dumps(events, ensure_ascii=False),
            "trajectory": [
                {"role": "user", "content": context + "\n\n请提取上述文本中的时间线信息。"},
                {"role": "assistant", "content": json.dumps(events, ensure_ascii=False)}
            ],
            "events": events
        }
        
        data.append(sft_item)
    
    return data


if __name__ == "__main__":
    # 下载约 1000 条数据
    dataset = download_thyme_sft(num_samples=1000)
    print(f"已生成 {len(dataset)} 条样本")
