"""
SFT 轨迹格式到 RL 问答格式数据的转换器
将 Thyme-SFT 数据集中的问答对转换为适合 RL 训练的格式
"""
import json
import random
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from transformers import AutoTokenizer


@dataclass
class RLDataItem:
    """RL 训练数据项"""
    id: str
    query: str                          # 用户查询/问题
    context: str                        # 上下文（医学报告）
    reference_answer: str                 # 参考答案（ground truth）
    system_prompt: str                  # 系统提示
    
    # RL 相关字段
    prompt: str                         # 完整提示（system + context + query）
    prompt_ids: List[int]               # tokenized prompt
    reference_ids: Optional[List[int]]  # tokenized reference answer
    
    # 奖励计算相关
    events: List[Dict[str, str]]        # 结构化事件信息（用于奖励计算）
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SFTtoRLConverter:
    """
    将 SFT 格式数据转换为 RL 训练格式
    
    SFT 格式:
    {
        "trajectory": [
            {"role": "user", "content": ...},
            {"role": "assistant", "content": ...}
        ],
        "context": ...,
        "output": ...
    }
    
    RL 格式:
    {
        "query": ...,          # 问题/指令
        "context": ...,        # 背景信息
        "reference": ...,      # 参考答案
        "prompt": ...,         # 完整提示
        "reward_fn": ...      # 奖励计算函数输入
    }
    """
    
    def __init__(
        self,
        tokenizer_name: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        max_prompt_length: int = 2048,
        max_response_length: int = 512
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            trust_remote_code=True,
            padding_side="left"
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.max_prompt_length = max_prompt_length
        self.max_response_length = max_response_length
        
        # 系统提示模板
        self.system_prompt = """你是一个医学时间线提取助手。你的任务是从医学报告中提取关键事件及其发生时间。
请仔细阅读提供的医学文本，识别所有时间相关的事件（如入院、诊断、手术、用药等），并以 JSON 格式输出。
输出格式：
[
  {"event": "事件描述", "date": "发生时间", "type": "事件类型"},
  ...
]
请确保：
1. 只输出 JSON 格式的结果，不要有其他内容
2. 准确提取时间和事件
3. 时间类型包括: admission, discharge, diagnosis, procedure, medication_start, symptom_onset 等"""

    def convert_sft_to_rl(
        self,
        sft_data: List[Dict[str, Any]],
        output_path: Optional[str] = None
    ) -> List[RLDataItem]:
        """
        将 SFT 数据转换为 RL 格式
        """
        rl_data = []
        
        for item in sft_data:
            try:
                rl_item = self._convert_single(item)
                if rl_item:
                    rl_data.append(rl_item)
            except Exception as e:
                print(f"转换样本 {item.get('id', 'unknown')} 失败: {e}")
                continue
        
        # 保存转换后的数据
        if output_path:
            self._save_rl_data(rl_data, output_path)
        
        return rl_data
    
    def _convert_single(self, sft_item: Dict[str, Any]) -> Optional[RLDataItem]:
        """
        转换单个 SFT 样本到 RL 格式
        """
        item_id = sft_item.get("id", f"item_{random.randint(0, 999999)}")
        context = sft_item.get("context", "")
        trajectory = sft_item.get("trajectory", [])
        events = sft_item.get("events", [])
        
        if len(trajectory) < 2:
            return None
        
        # 提取用户查询和参考答案
        user_msg = trajectory[0]
        assistant_msg = trajectory[1]
        
        query = user_msg.get("content", "")
        reference = assistant_msg.get("content", "")
        
        # 构建完整提示（Qwen 格式）
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"医学报告：\n{context}\n\n请提取上述文本中的时间线信息。"}
        ]
        
        # 使用 chat template 构建 prompt
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Tokenize
        prompt_ids = self.tokenizer.encode(
            prompt,
            max_length=self.max_prompt_length,
            truncation=True
        )
        
        reference_ids = self.tokenizer.encode(
            reference,
            max_length=self.max_response_length,
            truncation=True
        )
        
        return RLDataItem(
            id=item_id,
            query=query,
            context=context,
            reference_answer=reference,
            system_prompt=self.system_prompt,
            prompt=prompt,
            prompt_ids=prompt_ids,
            reference_ids=reference_ids,
            events=events
        )
    
    def _save_rl_data(self, rl_data: List[RLDataItem], output_path: str):
        """
        保存 RL 格式数据
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 保存为 JSONL 格式
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in rl_data:
                f.write(json.dumps(item.to_dict(), ensure_ascii=False) + '\n')
        
        print(f"RL 格式数据已保存到: {output_path}")
        
        # 同时保存为 HuggingFace datasets 格式
        try:
            from datasets import Dataset
            dataset = Dataset.from_list([item.to_dict() for item in rl_data])
            dataset.save_to_disk(str(output_path.with_suffix('')))
            print(f"Dataset 格式已保存到: {output_path.with_suffix('')}")
        except Exception as e:
            print(f"保存 Dataset 格式失败: {e}")


class RewardFunction:
    """
    医学时间线提取任务的奖励函数
    """
    
    def __init__(self):
        pass
    
    def compute_reward(
        self,
        generated_answer: str,
        reference_events: List[Dict[str, str]],
        generated_text: str = None
    ) -> Dict[str, float]:
        """
        计算生成答案的奖励
        返回多个维度的奖励分数
        """
        rewards = {}
        
        # 1. 格式奖励：是否正确输出 JSON
        rewards["format"] = self._format_reward(generated_answer)
        
        # 2. 事件提取奖励：提取的事件数量准确性
        rewards["event_coverage"] = self._event_coverage_reward(
            generated_answer, reference_events
        )
        
        # 3. 时间准确性奖励：时间信息是否正确
        rewards["temporal_accuracy"] = self._temporal_accuracy_reward(
            generated_answer, reference_events
        )
        
        # 4. 总体奖励（加权和）
        rewards["total"] = (
            0.2 * rewards["format"] +
            0.4 * rewards["event_coverage"] +
            0.4 * rewards["temporal_accuracy"]
        )
        
        return rewards
    
    def _format_reward(self, answer: str) -> float:
        """
        检查输出是否为有效的 JSON 格式
        """
        import json
        try:
            data = json.loads(answer)
            if isinstance(data, list):
                return 1.0
            return 0.5
        except:
            # 尝试提取 JSON 部分
            try:
                start = answer.find('[')
                end = answer.rfind(']')
                if start != -1 and end != -1:
                    json_str = answer[start:end+1]
                    data = json.loads(json_str)
                    if isinstance(data, list):
                        return 0.5  # 部分正确，有额外文本
            except:
                pass
            return 0.0
    
    def _event_coverage_reward(
        self,
        answer: str,
        reference_events: List[Dict[str, str]]
    ) -> float:
        """
        计算事件覆盖度奖励
        """
        import json
        
        try:
            # 解析生成的答案
            start = answer.find('[')
            end = answer.rfind(']')
            if start == -1 or end == -1:
                return 0.0
            
            json_str = answer[start:end+1]
            generated_events = json.loads(json_str)
            
            if not isinstance(generated_events, list):
                return 0.0
            
            # 计算召回率
            ref_events_set = set()
            for event in reference_events:
                key = (event.get("event", ""), event.get("type", ""))
                ref_events_set.add(key)
            
            matched = 0
            for gen_event in generated_events:
                key = (gen_event.get("event", ""), gen_event.get("type", ""))
                if key in ref_events_set:
                    matched += 1
            
            if len(ref_events_set) == 0:
                return 1.0 if len(generated_events) == 0 else 0.0
            
            recall = matched / len(ref_events_set)
            precision = matched / max(len(generated_events), 1)
            
            # F1 score
            if recall + precision == 0:
                return 0.0
            f1 = 2 * recall * precision / (recall + precision)
            
            return f1
            
        except Exception as e:
            return 0.0
    
    def _temporal_accuracy_reward(
        self,
        answer: str,
        reference_events: List[Dict[str, str]]
    ) -> float:
        """
        计算时间准确性奖励
        """
        import json
        
        try:
            start = answer.find('[')
            end = answer.rfind(']')
            if start == -1 or end == -1:
                return 0.0
            
            json_str = answer[start:end+1]
            generated_events = json.loads(json_str)
            
            if not isinstance(generated_events, list):
                return 0.0
            
            # 构建参考事件的字典（用于快速查找）
            ref_dict = {}
            for event in reference_events:
                key = event.get("event", "")
                ref_dict[key] = event.get("date", "")
            
            # 计算时间匹配率
            correct = 0
            total = 0
            
            for gen_event in generated_events:
                event_name = gen_event.get("event", "")
                event_date = gen_event.get("date", "")
                
                if event_name in ref_dict:
                    total += 1
                    if event_date == ref_dict[event_name]:
                        correct += 1
            
            if total == 0:
                return 0.0
            
            return correct / total
            
        except Exception as e:
            return 0.0


def main():
    """
    示例：转换数据
    """
    # 加载 SFT 数据
    from data_download import generate_timeline_data
    
    print("生成 SFT 数据...")
    sft_data = generate_timeline_data(num_samples=1000)
    
    # 保存 SFT 数据
    with open("data/thyme_sft.json", 'w', encoding='utf-8') as f:
        json.dump(sft_data, f, ensure_ascii=False, indent=2)
    
    print(f"SFT 数据已保存: {len(sft_data)} 条")
    
    # 转换为 RL 格式
    print("转换为 RL 格式...")
    converter = SFTtoRLConverter()
    rl_data = converter.convert_sft_to_rl(
        sft_data,
        output_path="data/thyme_rl.jsonl"
    )
    
    print(f"转换完成: {len(rl_data)} 条")
    
    # 测试奖励函数
    print("\n测试奖励函数...")
    reward_fn = RewardFunction()
    
    test_answer = '[{"event": "入院", "date": "2023年1月1日", "type": "admission"}]'
    test_events = [{"event": "入院", "date": "2023年1月1日", "type": "admission"}]
    
    rewards = reward_fn.compute_reward(test_answer, test_events)
    print(f"奖励分数: {rewards}")


if __name__ == "__main__":
    main()
