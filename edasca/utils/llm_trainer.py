"""
大模型情感分析模块
"""

import json
import numpy as np
from typing import Dict, Tuple, Optional
import openai
from dataclasses import dataclass
import requests
import time
import os

from ..utils.env_config import Config


@dataclass
class EmotionAnalysis:
    """情感分析结果"""
    pleasure: float  # 愉悦度 [-1, 1]
    arousal: float  # 唤醒度 [-1, 1]
    dominance: float  # 支配度 [-1, 1]
    emotion_label: str  # 情感标签
    confidence: float  # 置信度 [0, 1]


class LLMEmotionAnalyzer:
    """基于大模型的情感分析器"""
    
    def __init__(self):
        self.api_key = Config.OPENAI_API_KEY
        self.base_url = Config.OPENAI_BASE_URL
        self.model = Config.OPENAI_MODEL
        
        # 尝试初始化OpenAI客户端
        try:
            self.client = openai.OpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
            self.use_direct_api = False
        except Exception:
            self.use_direct_api = True
    
    def analyze_dialogue(self, dialogue: Dict[str, str]) -> Optional[EmotionAnalysis]:
        """分析对话的情感"""
        try:
            # 构建提示
            prompt = f"{Config.EMOTION_ANALYSIS_PROMPT}\n\n"
            prompt += f"用户输入: {dialogue['input']}\n"
            prompt += f"系统回复: {dialogue['output']}\n\n"
            prompt += "请以JSON格式返回，包含以下字段：\n"
            prompt += "- pleasure: 愉悦度数值\n"
            prompt += "- arousal: 唤醒度数值\n"
            prompt += "- dominance: 支配度数值\n"
            prompt += "- emotion_label: 情感标签\n"
            prompt += "- confidence: 置信度\n"
            
            # 调用大模型
            if self.use_direct_api:
                response = self._direct_api_call(prompt)
            else:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "你是一个专业的情感分析专家。"},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    timeout=30  # 添加超时设置
                )
            
            # 解析响应
            result_text = response.choices[0].message.content
            
            # 尝试提取JSON
            try:
                # 查找JSON部分
                start = result_text.find('{')
                end = result_text.rfind('}') + 1
                if start != -1 and end != -1:
                    json_str = result_text[start:end]
                    result = json.loads(json_str)
                    
                    return EmotionAnalysis(
                        pleasure=float(result.get('pleasure', 0.0)),
                        arousal=float(result.get('arousal', 0.0)),
                        dominance=float(result.get('dominance', 0.0)),
                        emotion_label=result.get('emotion_label', 'neutral'),
                        confidence=float(result.get('confidence', 0.5))
                    )
            except json.JSONDecodeError:
                pass
            
            # 如果JSON解析失败，尝试简单解析
            return self._parse_simple_response(result_text)
            
        except Exception as e:
            print(f"情感分析失败: {e}")
            
            # 如果API调用失败，使用简单的规则分析
            return self._rule_based_analysis(dialogue)
    
    def _direct_api_call(self, prompt: str):
        """直接API调用（备选方案）"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "你是一个专业的情感分析专家。"},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=data,
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"直接API调用失败: {e}")
            raise e
    
    def _parse_simple_response(self, text: str) -> EmotionAnalysis:
        """简单解析响应"""
        # 默认值
        pleasure = 0.0
        arousal = 0.0
        dominance = 0.0
        emotion_label = "neutral"
        confidence = 0.5
        
        # 简单的关键词匹配
        positive_words = ['开心', '快乐', '高兴', '喜欢', '满意']
        negative_words = ['难过', '伤心', '生气', '讨厌', '失望']
        
        if any(word in text for word in positive_words):
            pleasure = 0.5
            emotion_label = "happy"
        elif any(word in text for word in negative_words):
            pleasure = -0.5
            emotion_label = "sad"
        
        return EmotionAnalysis(
            pleasure=pleasure,
            arousal=arousal,
            dominance=dominance,
            emotion_label=emotion_label,
            confidence=confidence
        )
    
    def _rule_based_analysis(self, dialogue: Dict[str, str]) -> EmotionAnalysis:
        """基于规则的情感分析（API失败时的备选方案）"""
        # 默认值
        pleasure = 0.0
        arousal = 0.0
        dominance = 0.0
        emotion_label = "neutral"
        confidence = 0.5
        
        # 分析对话内容
        input_text = dialogue['input'].lower()
        output_text = dialogue['output'].lower()
        
        # 正面关键词
        positive_indicators = ['是', '对', '正确', '是的', '没错', '好的', '喜欢', '开心']
        # 负面关键词
        negative_indicators = ['不', '不是', '不对', '错误', '错了', '难过', '生气']
        
        # 检查输入和输出
        all_text = input_text + " " + output_text
        
        positive_count = sum(1 for word in positive_indicators if word in all_text)
        negative_count = sum(1 for word in negative_indicators if word in all_text)
        
        # 计算情感值
        if positive_count > negative_count:
            pleasure = 0.3
            emotion_label = "happy"
            confidence = 0.7
        elif negative_count > positive_count:
            pleasure = -0.3
            emotion_label = "sad"
            confidence = 0.7
        elif '正确' in all_text or '对' in all_text:
            pleasure = 0.2
            emotion_label = "calm"
            confidence = 0.6
        elif '错误' in all_text or '不对' in all_text:
            pleasure = -0.2
            emotion_label = "disappointed"
            confidence = 0.6
        
        return EmotionAnalysis(
            pleasure=pleasure,
            arousal=arousal,
            dominance=dominance,
            emotion_label=emotion_label,
            confidence=confidence
        )


class DialogueTrainer:
    """对话训练器"""
    
    def __init__(self):
        self.emotion_analyzer = LLMEmotionAnalyzer()
    
    def load_dialogue_file(self, filepath: str) -> list:
        """加载对话文件"""
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def train_single_dialogue(self, dialogue: Dict[str, str], edasca_instance) -> bool:
        """训练单个对话"""
        print(f"\n训练对话: {dialogue['input'][:30]}...")
        
        # 1. 情感分析
        emotion_analysis = self.emotion_analyzer.analyze_dialogue(dialogue)
        if not emotion_analysis:
            print("情感分析失败，跳过")
            return False
        
        print(f"情感分析: {emotion_analysis.emotion_label} "
              f"(P={emotion_analysis.pleasure:.2f}, "
              f"A={emotion_analysis.arousal:.2f}, "
              f"D={emotion_analysis.dominance:.2f})")
        
        # 2. 处理用户输入
        edasca_instance.process_input(dialogue['input'])
        
        # 3. 应用情感
        edasca_instance.emotion_system.apply_emotional_impact(
            np.array([emotion_analysis.pleasure, 
                     emotion_analysis.arousal, 
                     emotion_analysis.dominance])
        )
        
        # 4. 处理系统回复
        from edasca.core.data_structures import Origin
        edasca_instance.process_input(dialogue['output'], origin=Origin.INTERNAL)
        
        # 5. 等待处理完成
        import time
        time.sleep(1)
        
        return True
    
    def train_from_file(self, filepath: str, edasca_instance, save_interval: int = 1) -> str:
        """从文件训练对话"""
        # 加载对话数据
        dialogues = self.load_dialogue_file(filepath)
        print(f"加载了 {len(dialogues)} 条对话")
        
        # 创建保存目录
        save_dir = "model/"  # 固定保存到model目录
        os.makedirs(save_dir, exist_ok=True)
        
        # 训练统计
        total_success_count = 0
        round_num = 1
        
        # 尝试加载之前的轮数
        model_path = os.path.join(save_dir, "model.pkl")
        if os.path.exists(model_path):
            try:
                # 先尝试加载模型获取轮数
                temp_edasca = type(edasca_instance)()
                temp_edasca.load_state(model_path)
                training_info = temp_edasca.get_training_info()
                round_num = training_info['training_round'] + 1
                total_success_count = training_info.get('total_success_count', 0)
                print(f"从现有模型恢复：当前轮数 {round_num-1}，累计成功 {total_success_count}")
            except Exception as e:
                print(f"无法从模型恢复轮数: {e}，从第1轮开始")
                round_num = 1
                total_success_count = 0
        else:
            round_num = 1
            total_success_count = 0
        
        while True:  # 无限循环，直到手动停止
            print(f"\n{'='*50}")
            print(f"开始第 {round_num} 轮训练")
            print(f"{'='*50}")
            
            round_success_count = 0
            total_count = len(dialogues)
            
            # 逐条训练
            for i, dialogue in enumerate(dialogues, 1):
                print(f"\n=== 第{round_num}轮 - 进度: {i}/{total_count} ===")
                
                # 训练单条对话
                if self.train_single_dialogue(dialogue, edasca_instance):
                    round_success_count += 1
                    total_success_count += 1
                
                # 更新EDASCA实例的轮数信息
                edasca_instance.training_round = round_num
                edasca_instance.dialogue_index = i
                edasca_instance.total_success_count = total_success_count
                
                # 覆盖保存到model.pkl
                model_path = os.path.join(save_dir, "model.pkl")
                edasca_instance.save_state(model_path)
                print(f"模型已覆盖保存到: {model_path}")
                
                # 短暂暂停，避免过快保存
                time.sleep(0.5)
            
            # 本轮训练完成
            print(f"\n第 {round_num} 轮训练完成！")
            print(f"本轮成功: {round_success_count}/{total_count}")
            print(f"总成功数: {total_success_count}")
            
            # 保存本轮轮数模型（不覆盖）
            round_model_path = os.path.join(save_dir, f"model_round{round_num}.pkl")
            edasca_instance.save_state(round_model_path)
            print(f"第{round_num}轮模型已保存到: {round_model_path}")
            
            # 生成累计报告
            report = f"""
累计训练报告 - 第 {round_num} 轮完成
========================================
每轮对话数: {total_count}
当前轮成功: {round_success_count}/{total_count}
累计成功数: {total_success_count}
累计训练轮数: {round_num}
当前模型轮数: {round_num}
最新模型: {model_path}
轮数备份: {round_model_path}
            """
            
            # 保存报告
            report_path = os.path.join(save_dir, "training_report.txt")
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report)
            
            # 准备下一轮
            round_num += 1
            print(f"\n等待3秒后开始下一轮训练...")
            print("按 Ctrl+C 停止训练")
            time.sleep(3)