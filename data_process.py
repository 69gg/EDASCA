"""
数据预处理脚本
提前分析对话的情感等信息，保存到processed_dialogue.json
支持RPM设置和多线程处理
"""

import json
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
import os
import time
import threading
from queue import Queue
from concurrent.futures import ThreadPoolExecutor, as_completed

# 添加项目路径
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from edasca.utils.llm_trainer import LLMEmotionAnalyzer, EmotionAnalysis
from edasca.utils.env_config import Config


class DataProcessor:
    """数据预处理器"""
    
    def __init__(self, rpm: int = 60, max_workers: int = 4):
        """
        初始化数据预处理器
        
        Args:
            rpm: 每分钟请求数限制
            max_workers: 最大线程数
        """
        self.emotion_analyzer = LLMEmotionAnalyzer()
        self.processed_data = []
        self.rpm = rpm
        self.max_workers = max_workers
        self.request_interval = 60.0 / rpm if rpm > 0 else 0
        self.last_request_time = 0
        self.request_lock = threading.Lock()
    
    def _wait_for_rate_limit(self):
        """等待以符合RPM限制"""
        if self.rpm <= 0:
            return
        
        with self.request_lock:
            current_time = time.time()
            time_since_last_request = current_time - self.last_request_time
            
            if time_since_last_request < self.request_interval:
                sleep_time = self.request_interval - time_since_last_request
                time.sleep(sleep_time)
            
            self.last_request_time = time.time()
    
    def _process_single_dialogue(self, dialogue: Dict, index: int) -> Dict:
        """处理单个对话（线程安全）"""
        # 等待RPM限制
        self._wait_for_rate_limit()
        
        try:
            # 分析情感
            emotion_analysis = self.emotion_analyzer.analyze_dialogue(dialogue)
            
            if emotion_analysis:
                # 创建处理后的数据
                processed_dialogue = {
                    'input': dialogue['input'],
                    'output': dialogue['output'],
                    # 情感分析结果
                    'emotion': {
                        'pleasure': emotion_analysis.pleasure,
                        'arousal': emotion_analysis.arousal,
                        'dominance': emotion_analysis.dominance,
                        'emotion_label': emotion_analysis.emotion_label,
                        'confidence': emotion_analysis.confidence
                    },
                    # 文本特征
                    'text_features': {
                        'input_length': len(dialogue['input']),
                        'output_length': len(dialogue['output']),
                        'input_word_count': len(dialogue['input'].split()),
                        'output_word_count': len(dialogue['output'].split())
                    },
                    # 处理时间戳
                    'processed_at': datetime.now().isoformat()
                }
                return processed_dialogue, True
            else:
                # 如果情感分析失败，使用默认值
                processed_dialogue = {
                    'input': dialogue['input'],
                    'output': dialogue['output'],
                    'emotion': {
                        'pleasure': 0.0,
                        'arousal': 0.0,
                        'dominance': 0.0,
                        'emotion_label': 'neutral',
                        'confidence': 0.5
                    },
                    'text_features': {
                        'input_length': len(dialogue['input']),
                        'output_length': len(dialogue['output']),
                        'input_word_count': len(dialogue['input'].split()),
                        'output_word_count': len(dialogue['output'].split())
                    },
                    'processed_at': datetime.now().isoformat(),
                    'analysis_failed': True
                }
                return processed_dialogue, False
                
        except Exception as e:
            print(f"\n处理第{index}条对话时出错: {e}")
            # 返回失败的数据
            processed_dialogue = {
                'input': dialogue['input'],
                'output': dialogue['output'],
                'emotion': {
                    'pleasure': 0.0,
                    'arousal': 0.0,
                    'dominance': 0.0,
                    'emotion_label': 'neutral',
                    'confidence': 0.5
                },
                'text_features': {
                    'input_length': len(dialogue['input']),
                    'output_length': len(dialogue['output']),
                    'input_word_count': len(dialogue['input'].split()),
                    'output_word_count': len(dialogue['output'].split())
                },
                'processed_at': datetime.now().isoformat(),
                'analysis_failed': True,
                'error': str(e)
            }
            return processed_dialogue, False
    
    def process_dialogue_file(self, input_file: str = "dialogue.json", 
                             output_file: str = "dialogue_processed.json") -> None:
        """处理对话文件"""
        
        print("=== 数据预处理开始 ===")
        print(f"RPM限制: {self.rpm}")
        print(f"最大线程数: {self.max_workers}\n")
        
        # 加载原始对话数据
        print(f"加载对话文件: {input_file}")
        with open(input_file, 'r', encoding='utf-8') as f:
            dialogues = json.load(f)
        
        print(f"共加载 {len(dialogues)} 条对话\n")
        
        # 使用多线程处理
        processed_count = 0
        failed_count = 0
        results = []
        
        print(f"开始多线程处理...")
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有任务
            future_to_index = {
                executor.submit(self._process_single_dialogue, dialogue, i): (i, dialogue)
                for i, dialogue in enumerate(dialogues, 1)
            }
            
            # 处理完成的任务
            for future in as_completed(future_to_index):
                index, dialogue = future_to_index[future]
                try:
                    result, success = future.result()
                    results.append((index, result, success))
                    
                    if success:
                        processed_count += 1
                    else:
                        failed_count += 1
                    
                    # 显示进度
                    progress = len(results) / len(dialogues) * 100
                    print(f"\r处理进度: {len(results)}/{len(dialogues)} ({progress:.1f}%)", end="")
                    
                except Exception as e:
                    print(f"\n处理第{index}条对话时发生异常: {e}")
                    failed_count += 1
        
        # 按原始顺序排序结果
        results.sort(key=lambda x: x[0])
        self.processed_data = [result[1] for result in results]
        
        # 计算处理时间
        end_time = time.time()
        processing_time = end_time - start_time
        avg_time_per_dialogue = processing_time / len(dialogues)
        
        # 保存处理后的数据
        print(f"\n\n保存处理后的数据到: {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.processed_data, f, ensure_ascii=False, indent=2)
        
        # 生成统计报告
        self._generate_statistics_report(processed_count, failed_count, processing_time)
        
        print(f"\n=== 数据预处理完成 ===")
        print(f"成功处理: {processed_count}")
        print(f"失败数量: {failed_count}")
        print(f"总处理时间: {processing_time:.2f}秒")
        print(f"平均每条: {avg_time_per_dialogue:.2f}秒")
        print(f"输出文件: {output_file}")
    
    def _generate_statistics_report(self, processed_count: int, failed_count: int, processing_time: float = 0) -> None:
        """生成统计报告"""
        
        # 情感分布统计
        emotion_counts = {}
        pleasure_values = []
        arousal_values = []
        dominance_values = []
        
        for data in self.processed_data:
            emotion = data['emotion']
            label = emotion['emotion_label']
            
            emotion_counts[label] = emotion_counts.get(label, 0) + 1
            pleasure_values.append(emotion['pleasure'])
            arousal_values.append(emotion['arousal'])
            dominance_values.append(emotion['dominance'])
        
        # 文本长度统计
        input_lengths = [d['text_features']['input_length'] for d in self.processed_data]
        output_lengths = [d['text_features']['output_length'] for d in self.processed_data]
        
        # 生成报告
        report = f"""
数据预处理统计报告
===================
生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

基本情况
--------
总对话数: {len(self.processed_data)}
成功分析: {processed_count}
分析失败: {failed_count}
成功率: {processed_count/len(self.processed_data)*100:.1f}%
处理时间: {processing_time:.2f}秒
RPM设置: {self.rpm}
并发线程数: {self.max_workers}
平均处理速度: {len(self.processed_data)/processing_time*60:.1f} RPM

情感分布
--------
"""
        
        for emotion, count in sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = count / len(self.processed_data) * 100
            report += f"{emotion}: {count} ({percentage:.1f}%)\n"
        
        report += f"""
PAD数值统计
----------
愉悦度(P): 平均={np.mean(pleasure_values):.3f}, 标准差={np.std(pleasure_values):.3f}, 范围=[{np.min(pleasure_values):.3f}, {np.max(pleasure_values):.3f}]
唤醒度(A): 平均={np.mean(arousal_values):.3f}, 标准差={np.std(arousal_values):.3f}, 范围=[{np.min(arousal_values):.3f}, {np.max(arousal_values):.3f}]
支配度(D): 平均={np.mean(dominance_values):.3f}, 标准差={np.std(dominance_values):.3f}, 范围=[{np.min(dominance_values):.3f}, {np.max(dominance_values):.3f}]

文本长度统计
----------
输入长度: 平均={np.mean(input_lengths):.1f}, 中位数={np.median(input_lengths):.1f}, 最大={np.max(input_lengths)}
输出长度: 平均={np.mean(output_lengths):.1f}, 中位数={np.median(output_lengths):.1f}, 最大={np.max(output_lengths)}
"""
        
        # 保存报告
        report_file = "data_processing_report.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"\n统计报告已保存到: {report_file}")
        
        # 显示简要统计
        print("\n简要统计:")
        print(f"情感标签数量: {len(emotion_counts)}")
        print(f"主要情感: {max(emotion_counts, key=emotion_counts.get)}")
        print(f"平均愉悦度: {np.mean(pleasure_values):.3f}")


def create_fast_trainer():
    """创建一个快速训练器，使用预处理的数据"""
    
    # 这里可以创建一个新的训练器类，直接读取预处理的数据
    # 避免在训练时重复进行情感分析
    
    class FastDialogueTrainer:
        """快速对话训练器（使用预处理数据）"""
        
        def __init__(self, processed_file="dialogue_processed.json"):
            self.processed_file = processed_file
            self.processed_data = None
        
        def load_processed_data(self) -> List[Dict]:
            """加载预处理的数据"""
            if self.processed_data is None:
                with open(self.processed_file, 'r', encoding='utf-8') as f:
                    self.processed_data = json.load(f)
            return self.processed_data
        
        def train_single_dialogue(self, dialogue_data: Dict, edasca_instance) -> bool:
            """训练单个对话（使用预处理的数据）"""
            print(f"\n训练对话: {dialogue_data['input'][:30]}...")
            
            # 获取预处理的情感数据
            emotion_data = dialogue_data['emotion']
            
            print(f"使用预处理的情感: {emotion_data['emotion_label']} "
                  f"(P={emotion_data['pleasure']:.2f}, "
                  f"A={emotion_data['arousal']:.2f}, "
                  f"D={emotion_data['dominance']:.2f})")
            
            # 1. 处理用户输入
            edasca_instance.process_input(dialogue_data['input'])
            
            # 2. 应用情感
            edasca_instance.emotion_system.apply_emotional_impact(
                np.array([emotion_data['pleasure'], 
                         emotion_data['arousal'], 
                         emotion_data['dominance']])
            )
            
            # 3. 处理系统回复
            from edasca.core.data_structures import Origin
            edasca_instance.process_input(dialogue_data['output'], origin=Origin.INTERNAL)
            
            # 4. 等待处理完成
            import time
            time.sleep(1)
            
            return True
        
        def train_from_file(self, edasca_instance, save_interval: int = 1) -> str:
            """从预处理文件训练"""
            # 加载数据
            dialogues = self.load_processed_data()
            print(f"加载了 {len(dialogues)} 条预处理对话")
            
            # 创建保存目录
            save_dir = "model/"
            os.makedirs(save_dir, exist_ok=True)
            
            # 训练统计
            total_success_count = 0
            round_num = 1
            
            # 尝试加载之前的轮数
            model_path = os.path.join(save_dir, "model.pkl")
            if os.path.exists(model_path):
                try:
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
    
    return FastDialogueTrainer


if __name__ == "__main__":
    import argparse
    
    # 设置命令行参数
    parser = argparse.ArgumentParser(description='EDASCA数据预处理工具')
    parser.add_argument('--rpm', type=int, default=60, 
                       help='每分钟请求数限制 (默认: 60)')
    parser.add_argument('--workers', type=int, default=4,
                       help='并发线程数 (默认: 4)')
    parser.add_argument('--input', type=str, default='dialogue.json',
                       help='输入文件路径 (默认: dialogue.json)')
    parser.add_argument('--output', type=str, default='dialogue_processed.json',
                       help='输出文件路径 (默认: dialogue_processed.json)')
    
    args = parser.parse_args()
    
    # 运行数据预处理
    print(f"使用参数: RPM={args.rpm}, 并发数={args.workers}")
    processor = DataProcessor(rpm=args.rpm, max_workers=args.workers)
    processor.process_dialogue_file(args.input, args.output)
    
    print("\n预处理完成！现在可以使用 train.py 进行训练了")