"""
模型评估脚本
用于评估训练好的 PPO/GRPO 模型
"""
import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from data_converter import RewardFunction
from data_download import generate_timeline_data


def load_model(model_path: str, base_model: str = "Qwen/Qwen2.5-VL-3B-Instruct"):
    """
    加载训练好的模型
    
    Args:
        model_path: LoRA adapter 或完整模型的路径
        base_model: 基础模型名称（如果使用 LoRA）
    
    Returns:
        model, tokenizer
    """
    print(f"加载模型: {model_path}")
    
    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path if os.path.exists(os.path.join(model_path, "tokenizer_config.json")) else base_model,
        trust_remote_code=True,
        padding_side="left"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 加载基础模型
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True
    )
    
    # 检查是否是 LoRA 模型
    if os.path.exists(os.path.join(model_path, "adapter_config.json")):
        print("加载 LoRA adapter...")
        model = PeftModel.from_pretrained(model, model_path)
    elif os.path.exists(os.path.join(model_path, "pytorch_model.bin")) or \
         os.path.exists(os.path.join(model_path, "model.safetensors")):
        print("加载完整模型...")
        # 直接加载模型权重
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True
        )
    
    model.eval()
    return model, tokenizer


def generate_response(
    model,
    tokenizer,
    prompt: str,
    max_length: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9
) -> str:
    """
    生成响应
    """
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # 解码，去掉 prompt 部分
    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )
    
    return response


def evaluate_model(
    model,
    tokenizer,
    test_data: List[Dict[str, Any]],
    num_samples: int = 100
) -> Dict[str, float]:
    """
    评估模型性能
    """
    reward_fn = RewardFunction()
    
    # 限制评估样本数
    test_data = test_data[:num_samples]
    
    results = {
        "format_rewards": [],
        "coverage_rewards": [],
        "temporal_rewards": [],
        "total_rewards": [],
        "responses": []
    }
    
    print(f"\n评估 {len(test_data)} 个样本...")
    
    for i, item in enumerate(test_data):
        print(f"处理样本 {i+1}/{len(test_data)}...", end="\r")
        
        # 生成响应
        response = generate_response(model, tokenizer, item["prompt"])
        
        # 计算奖励
        rewards = reward_fn.compute_reward(response, item.get("events", []))
        
        results["format_rewards"].append(rewards["format"])
        results["coverage_rewards"].append(rewards["event_coverage"])
        results["temporal_rewards"].append(rewards["temporal_accuracy"])
        results["total_rewards"].append(rewards["total"])
        
        # 保存部分响应用于展示
        if i < 5:
            results["responses"].append({
                "prompt": item["context"][:100] + "...",
                "response": response[:200] + "..." if len(response) > 200 else response,
                "reference": item["reference_answer"][:200] + "..." if len(item["reference_answer"]) > 200 else item["reference_answer"],
                "rewards": rewards
            })
    
    # 计算平均值
    summary = {
        "format_reward": sum(results["format_rewards"]) / len(results["format_rewards"]),
        "coverage_reward": sum(results["coverage_rewards"]) / len(results["coverage_rewards"]),
        "temporal_reward": sum(results["temporal_rewards"]) / len(results["temporal_rewards"]),
        "total_reward": sum(results["total_rewards"]) / len(results["total_rewards"]),
        "num_samples": len(test_data)
    }
    
    return summary, results


def main():
    parser = argparse.ArgumentParser(description="模型评估")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="训练好的模型路径"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="Qwen/Qwen2.5-VL-3B-Instruct",
        help="基础模型名称"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/thyme_rl.jsonl",
        help="测试数据路径"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=100,
        help="评估样本数"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="评估结果输出路径"
    )
    
    args = parser.parse_args()
    
    # 加载模型
    model, tokenizer = load_model(args.model_path, args.base_model)
    
    # 加载数据
    if os.path.exists(args.data_path):
        print(f"\n加载测试数据: {args.data_path}")
        test_data = []
        with open(args.data_path, 'r', encoding='utf-8') as f:
            for line in f:
                test_data.append(json.loads(line))
    else:
        print(f"\n测试数据不存在，生成 {args.num_samples} 条测试数据...")
        sft_data = generate_timeline_data(num_samples=args.num_samples)
        # 简单转换为 RL 格式
        from data_converter import SFTtoRLConverter
        converter = SFTtoRLConverter()
        rl_data = converter.convert_sft_to_rl(sft_data)
        test_data = [item.to_dict() for item in rl_data]
    
    # 评估
    summary, results = evaluate_model(
        model, tokenizer, test_data, args.num_samples
    )
    
    # 打印结果
    print("\n" + "=" * 50)
    print("评估结果:")
    print("=" * 50)
    print(f"评估样本数: {summary['num_samples']}")
    print(f"格式奖励: {summary['format_reward']:.4f}")
    print(f"事件覆盖奖励: {summary['coverage_reward']:.4f}")
    print(f"时间准确性奖励: {summary['temporal_reward']:.4f}")
    print(f"总奖励: {summary['total_reward']:.4f}")
    print("=" * 50)
    
    # 展示部分响应
    print("\n示例响应:")
    print("-" * 50)
    for i, resp in enumerate(results["responses"]):
        print(f"\n示例 {i+1}:")
        print(f"提示: {resp['prompt']}")
        print(f"生成: {resp['response']}")
        print(f"参考: {resp['reference']}")
        print(f"奖励: {resp['rewards']}")
        print("-" * 50)
    
    # 保存结果
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({
                "summary": summary,
                "results": results
            }, f, ensure_ascii=False, indent=2)
        
        print(f"\n结果已保存到: {output_path}")


if __name__ == "__main__":
    main()
