"""
SFT 轨迹格式到 RL 问答格式数据的转换器
针对 GQA 图像问答数据集 (Thyme-SFT 格式变体)
"""
import json
import re
import random
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import pandas as pd
import numpy as np


try:
    from transformers import AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("警告: transformers 未安装，将使用简单文本处理")


@dataclass
class RLDataItem:
    """RL 训练数据项"""
    id: str
    query: str                          # 用户问题（纯文本，不含格式模板）
    images: Optional[List[str]]         # base64 图像数据列表
    reference_answer: str               # 参考答案（从<answer>标签提取的纯答案）
    full_response: str                  # 完整回答（含推理过程，用于训练）
    system_prompt: str                  # 系统提示
    
    # RL 相关字段
    prompt: str                         # 完整提示（system + query + 格式要求）
    prompt_ids: List[int]               # tokenized prompt
    reference_ids: Optional[List[int]]  # tokenized reference answer
    
    # 奖励计算相关
    answer_type: str                    # 问题类型（位置/判断/描述等）
    ground_truth: str                   # 标准答案（用于奖励计算）
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class GQASFTtoRLConverter:
    """
    将 GQA SFT 格式数据转换为 RL 训练格式
    
    SFT 格式:
    {
        "image": [base64_image],           # 图像数据
        "question": "<image>\n问题\n### Output Format...",  # 带格式模板的问题
        "response": "推理过程...<answer>答案</answer>"       # 完整回答
    }
    
    RL 格式:
    {
        "query": ...,              # 纯问题
        "images": ...,             # 图像数据
        "reference": ...,          # 参考答案
        "prompt": ...,             # 完整提示（给模型输入）
        "reward_fn_input": ...    # 奖励计算输入
    }
    """
    
    def __init__(
        self,
        tokenizer_name: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        max_prompt_length: int = 2048,
        max_response_length: int = 4096,
        use_tokenizer: bool = True
    ):
        self.tokenizer = None
        self.use_tokenizer = use_tokenizer and TRANSFORMERS_AVAILABLE
        
        if self.use_tokenizer:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    tokenizer_name,
                    trust_remote_code=True,
                    padding_side="left",
                    local_files_only=False
                )
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                print(f"Tokenizer 加载成功: {tokenizer_name}")
            except Exception as e:
                print(f"Tokenizer 加载失败: {e}")
                print("将使用简单文本分割代替 tokenization")
                self.tokenizer = None
            
        self.max_prompt_length = max_prompt_length
        self.max_response_length = max_response_length
        
        # 系统提示模板 - 针对 GQA 图像问答任务
        self.system_prompt = """你是一个图像问答助手。用户会提供一张图片并提问，你需要：
1. 仔细观察图片内容
2. 分析问题要点
3. 给出详细的推理过程（如果需要）
4. 最终以 <answer>答案</answer> 格式输出最终答案

请确保：
- 推理过程清晰、逻辑严密
- 最终答案放在 <answer></answer> 标签内
- 如果涉及位置判断，明确说明左/右/上/下等方位"""

    def _build_prompt_manual(self, query: str) -> str:
        """手动构建 Qwen-VL 格式 prompt"""
        prompt = "<|im_start|>system\n" + self.system_prompt + "\n<|im_end|>\n"
        prompt += "<|im_start|>user\n<image>\n" + query + "\n<|im_end|>\n"
        prompt += "<|im_start|>assistant\n"
        return prompt

    def extract_pure_question(self, question_text: str) -> str:
        """
        从带格式模板的问题中提取纯问题文本
        
        输入示例:
        <image>
        Which side is the tray on?
        
        ### User Image Path:** "..."
        ### User Image Size:** "..."
        ### **Output Format (strict adherence required):**
        ...
        
        输出:
        Which side is the tray on?
        """
        # 移除 <image> 标签
        text = re.sub(r'<image>', '', question_text)
        
        # 提取第一个句子/问题（通常是实际的问题）
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        # 找到第一个不是元数据的问题行
        for line in lines:
            if not line.startswith('###') and not line.startswith('User'):
                # 清理特殊字符
                question = line.strip()
                if question and not question.startswith('<') and len(question) > 10:
                    return question
        
        # 如果没找到，返回第一行非空行
        for line in lines:
            if line and not line.startswith('###') and not line.startswith('<') and len(line) > 10:
                return line
        
        return lines[0] if lines else "Unknown question"

    def extract_answer_from_response(self, response: str) -> Tuple[str, str, str]:
        """
        从 response 中提取各个部分
        
        返回:
        - reasoning: 推理过程（不含代码部分）
        - answer: <answer>标签内的纯答案
        - answer_type: 答案类型（位置/判断/描述等）
        """
        # 提取 <answer> 标签内的内容
        answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
        if answer_match:
            answer = answer_match.group(1).strip()
        else:
            # 如果没有 <answer> 标签，尝试提取最后一句话
            sentences = response.split('\n')
            answer = sentences[-1].strip() if sentences else response
        
        # 提取推理过程（<answer>之前的内容）
        reasoning_end = response.find('<answer>')
        if reasoning_end > 0:
            reasoning = response[:reasoning_end].strip()
            # 移除代码块
            reasoning = re.sub(r'<code>.*?</code>', '[代码部分]', reasoning, flags=re.DOTALL)
            reasoning = re.sub(r'<sandbox_output>.*?</sandbox_output>', '[输出]', reasoning, flags=re.DOTALL)
        else:
            reasoning = response
        
        # 判断答案类型
        answer_lower = answer.lower()
        if any(word in answer_lower for word in ['left', 'right', 'top', 'bottom', 'side']):
            answer_type = "位置判断"
        elif any(word in answer_lower for word in ['yes', 'no', '是', '否']):
            answer_type = "是非判断"
        elif any(word in answer_lower for word in ['what', 'where', 'which', 'who', 'how many']):
            answer_type = "事实问答"
        else:
            answer_type = "描述性回答"
        
        return reasoning, answer, answer_type

    def convert_sft_to_rl(
        self,
        sft_data: List[Dict[str, Any]],
        output_path: Optional[str] = None,
        sample_limit: Optional[int] = None
    ) -> List[RLDataItem]:
        """
        将 SFT 数据转换为 RL 格式
        
        Args:
            sft_data: SFT 格式数据列表
            output_path: 输出路径
            sample_limit: 限制转换的样本数（用于测试）
        """
        rl_data = []
        
        if sample_limit:
            sft_data = sft_data[:sample_limit]
        
        print(f"开始转换 {len(sft_data)} 条 SFT 数据...")
        
        for idx, item in enumerate(sft_data):
            try:
                rl_item = self._convert_single(item, idx)
                if rl_item:
                    rl_data.append(rl_item)
                    
                if (idx + 1) % 100 == 0:
                    print(f"已转换 {idx + 1}/{len(sft_data)} 条...")
                    
            except Exception as e:
                print(f"转换样本 {idx} 失败: {e}")
                continue
        
        print(f"转换完成: {len(rl_data)} 条有效数据")
        
        # 保存转换后的数据
        if output_path:
            self._save_rl_data(rl_data, output_path)
        
        return rl_data
    
    def _convert_single(self, sft_item: Dict[str, Any], idx: int) -> Optional[RLDataItem]:
        """
        转换单个 SFT 样本到 RL 格式
        """
        item_id = f"gqa_{idx:06d}"
        
        # 提取图像
        images = sft_item.get("image")
        if isinstance(images, np.ndarray):
            images = images.tolist()
        if not isinstance(images, list):
            images = [images] if images else []
        
        # 提取纯问题
        raw_question = sft_item.get("question", "")
        query = self.extract_pure_question(raw_question)
        
        # 提取答案信息
        raw_response = sft_item.get("response", "")
        reasoning, answer, answer_type = self.extract_answer_from_response(raw_response)
        
        # 构建 prompt 和 tokenize
        if self.tokenizer:
            # 构建 Qwen-VL 格式的 messages
            messages = [
                {"role": "system", "content": self.system_prompt},
                {
                    "role": "user", 
                    "content": [
                        {"type": "image", "image": "<image_placeholder>"},
                        {"type": "text", "text": query}
                    ]
                }
            ]
            
            try:
                prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            except Exception as e:
                prompt = self._build_prompt_manual(query)
            
            # Tokenize (文本部分)
            prompt_ids = self.tokenizer.encode(
                prompt,
                max_length=self.max_prompt_length,
                truncation=True
            )
            
            # 完整 response 作为参考（用于监督学习的 warmup）
            reference_ids = self.tokenizer.encode(
                raw_response,
                max_length=self.max_response_length,
                truncation=True
            )
        else:
            # 无 tokenizer 模式
            prompt = self._build_prompt_manual(query)
            # 使用简单的字符分割作为 placeholder
            prompt_ids = list(range(len(prompt) // 4))  # 粗略估计 token 数
            reference_ids = list(range(len(raw_response) // 4))
        
        return RLDataItem(
            id=item_id,
            query=query,
            images=images,
            reference_answer=answer,
            full_response=raw_response,
            system_prompt=self.system_prompt,
            prompt=prompt,
            prompt_ids=prompt_ids,
            reference_ids=reference_ids,
            answer_type=answer_type,
            ground_truth=answer
        )
    
    def _save_rl_data(self, rl_data: List[RLDataItem], output_path: str):
        """
        保存 RL 格式数据
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 保存为 JSONL 格式（不包含图像数据，太大）
        jsonl_path = output_path.with_suffix('.jsonl')
        with open(jsonl_path, 'w', encoding='utf-8') as f:
            for item in rl_data:
                item_dict = item.to_dict()
                # 图像数据太大，保存为引用
                if item_dict.get('images'):
                    item_dict['images'] = f"<{len(item_dict['images'])} images, base64 encoded>"
                f.write(json.dumps(item_dict, ensure_ascii=False) + '\n')
        
        print(f"RL 格式数据已保存到: {jsonl_path}")
        
        # 保存元数据（不含图像）
        metadata_path = output_path.parent / f"{output_path.stem}_metadata.json"
        metadata = []
        for item in rl_data:
            meta = {
                "id": item.id,
                "query": item.query,
                "reference_answer": item.reference_answer,
                "answer_type": item.answer_type,
                "prompt_length": len(item.prompt),
                "prompt_token_length": len(item.prompt_ids),
            }
            metadata.append(meta)
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        print(f"元数据已保存到: {metadata_path}")
        
        # 保存完整数据为 HuggingFace datasets 格式（pickle）
        try:
            import pickle
            full_path = output_path.parent / f"{output_path.stem}_full.pkl"
            with open(full_path, 'wb') as f:
                pickle.dump([item.to_dict() for item in rl_data], f)
            print(f"完整数据已保存到: {full_path}")
        except Exception as e:
            print(f"保存完整数据失败: {e}")


class GQARewardFunction:
    """
    GQA 图像问答任务的奖励函数
    
    基于答案正确性和格式的奖励计算
    """
    
    def __init__(self):
        pass
    
    def compute_reward(
        self,
        generated_answer: str,
        ground_truth: str,
        answer_type: str = "fact"
    ) -> Dict[str, float]:
        """
        计算生成答案的奖励
        
        Args:
            generated_answer: 模型生成的回答（含或不含<answer>标签）
            ground_truth: 标准答案
            answer_type: 答案类型（位置判断/是非判断/事实问答/描述性回答）
        
        返回:
            rewards: 各维度奖励分数
        """
        rewards = {}
        
        # 1. 格式奖励：是否正确使用 <answer> 标签
        rewards["format"] = self._format_reward(generated_answer)
        
        # 2. 答案提取与匹配奖励
        extracted_answer = self._extract_answer(generated_answer)
        rewards["answer_match"] = self._answer_match_reward(
            extracted_answer, ground_truth, answer_type
        )
        
        # 3. 推理过程奖励（长度和结构检查）
        rewards["reasoning"] = self._reasoning_reward(generated_answer)
        
        # 4. 总体奖励（加权和）
        rewards["total"] = (
            0.2 * rewards["format"] +
            0.5 * rewards["answer_match"] +
            0.3 * rewards["reasoning"]
        )
        
        return rewards
    
    def _format_reward(self, answer: str) -> float:
        """检查输出格式是否正确"""
        has_answer_tag = '<answer>' in answer and '</answer>' in answer
        
        if has_answer_tag:
            # 检查标签是否成对且正确嵌套
            answer_count = answer.count('<answer>')
            close_count = answer.count('</answer>')
            if answer_count == 1 and close_count == 1:
                return 1.0
            else:
                return 0.5  # 有标签但格式不完全正确
        else:
            # 没有 <answer> 标签
            return 0.1
    
    def _extract_answer(self, generated: str) -> str:
        """从生成文本中提取答案"""
        match = re.search(r'<answer>(.*?)</answer>', generated, re.DOTALL)
        if match:
            return match.group(1).strip()
        else:
            # 尝试提取最后一行作为答案
            lines = [l.strip() for l in generated.split('\n') if l.strip()]
            return lines[-1] if lines else generated
    
    def _answer_match_reward(
        self, 
        generated: str, 
        ground_truth: str,
        answer_type: str
    ) -> float:
        """计算答案匹配度"""
        gen_lower = generated.lower().strip()
        truth_lower = ground_truth.lower().strip()
        
        # 完全匹配
        if gen_lower == truth_lower:
            return 1.0
        
        # 根据答案类型的特殊处理
        if answer_type == "位置判断":
            # 检查位置关键词
            positions = ['left', 'right', 'top', 'bottom', 'center', 'middle']
            gen_pos = [p for p in positions if p in gen_lower]
            truth_pos = [p for p in positions if p in truth_lower]
            if gen_pos and truth_pos:
                return 0.5 if gen_pos[0] == truth_pos[0] else 0.0
        
        elif answer_type == "是非判断":
            # 检查 yes/no
            gen_yes = 'yes' in gen_lower or '是' in gen_lower
            gen_no = 'no' in gen_lower or '否' in gen_lower or 'not' in gen_lower
            truth_yes = 'yes' in truth_lower or '是' in truth_lower
            
            if (gen_yes and truth_yes) or (gen_no and not truth_yes):
                return 1.0
            elif (gen_yes and not truth_yes) or (gen_no and truth_yes):
                return 0.0
        
        # 包含关系检查
        if truth_lower in gen_lower or gen_lower in truth_lower:
            return 0.5
        
        # 关键词匹配
        gen_words = set(gen_lower.split())
        truth_words = set(truth_lower.split())
        if gen_words and truth_words:
            overlap = len(gen_words & truth_words)
            return 0.3 * (overlap / max(len(truth_words), 1))
        
        return 0.0
    
    def _reasoning_reward(self, answer: str) -> float:
        """评估推理过程质量"""
        # 移除代码块后检查推理文本长度
        reasoning_text = re.sub(r'<code>.*?</code>', '', answer, flags=re.DOTALL)
        reasoning_text = re.sub(r'<sandbox_output>.*?</sandbox_output>', '', reasoning_text, flags=re.DOTALL)
        
        # 检查是否有推理内容（<answer>之前的部分）
        answer_start = reasoning_text.find('<answer>')
        if answer_start > 0:
            reasoning_part = reasoning_text[:answer_start].strip()
            # 合理的推理长度（100-2000字符）
            length = len(reasoning_part)
            if 100 <= length <= 2000:
                return 1.0
            elif length > 2000:
                return 0.8  # 稍长但可接受
            elif length > 50:
                return 0.5  # 偏短
            else:
                return 0.2  # 太短
        
        return 0.0


def load_sft_data(parquet_path: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """从 parquet 文件加载 SFT 数据"""
    print(f"加载数据: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    
    if limit:
        df = df.head(limit)
    
    data = df.to_dict('records')
    print(f"加载完成: {len(data)} 条数据")
    return data


def main():
    """
    主函数：转换 GQA SFT 数据到 RL 格式
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='SFT to RL Data Converter')
    parser.add_argument('--input', type=str, 
                        default='/root/nju-rl/data/2round-00000-of-00069.parquet',
                        help='输入 parquet 文件路径')
    parser.add_argument('--output', type=str,
                        default='/root/nju-rl/data/gqa_rl',
                        help='输出文件前缀')
    parser.add_argument('--limit', type=int, default=1000,
                        help='转换的样本数限制')
    parser.add_argument('--tokenizer', type=str,
                        default='Qwen/Qwen2.5-VL-3B-Instruct',
                        help='Tokenizer 名称')
    parser.add_argument('--no-tokenizer', action='store_true',
                        help='不使用 tokenizer（离线模式）')
    
    args = parser.parse_args()
    
    # 加载 SFT 数据
    sft_data = load_sft_data(args.input, limit=args.limit)
    
    # 创建转换器
    print(f"\n初始化转换器...")
    converter = GQASFTtoRLConverter(
        tokenizer_name=args.tokenizer,
        use_tokenizer=not args.no_tokenizer
    )
    
    # 转换数据
    print(f"\n开始转换前 {args.limit} 条数据...")
    rl_data = converter.convert_sft_to_rl(
        sft_data,
        output_path=args.output
    )
    
    # 测试奖励函数
    print("\n\n测试奖励函数...")
    reward_fn = GQARewardFunction()
    
    # 用第一个样本测试
    if rl_data:
        sample = rl_data[0]
        print(f"\n样本: {sample.id}")
        print(f"问题: {sample.query}")
        print(f"答案类型: {sample.answer_type}")
        print(f"标准答案: {sample.reference_answer}")
        
        # 模拟生成回答
        test_generated = sample.full_response
        
        rewards = reward_fn.compute_reward(
            test_generated,
            sample.ground_truth,
            sample.answer_type
        )
        
        print(f"\n奖励分数:")
        for key, value in rewards.items():
            print(f"  {key}: {value:.3f}")
    
    # 打印统计信息
    print("\n\n转换后数据统计:")
    answer_types = {}
    for item in rl_data:
        t = item.answer_type
        answer_types[t] = answer_types.get(t, 0) + 1
    
    print("答案类型分布:")
    for t, count in sorted(answer_types.items(), key=lambda x: -x[1]):
        print(f"  {t}: {count}")
    
    print(f"\n完成! 共转换 {len(rl_data)} 条数据")


if __name__ == "__main__":
    main()
