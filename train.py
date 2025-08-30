"""
EDASCA训练脚本 - 支持对话格式训练
"""

import sys
import os
import json
import time
import numpy as np
from datetime import datetime
from typing import List, Dict, Any

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from edasca import EDASCA
from edasca.core.data_structures import Origin
from edasca.utils.llm_trainer import DialogueTrainer
from edasca.utils.env_config import Config


def _is_dialogue_file_processed(file_path: str) -> bool:
    """检查对话文件是否已经预处理
    
    通过检查JSON内容中的特定字段来判断是否已经预处理
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 如果是空文件，返回False
        if not data:
            return False
        
        # 取第一条数据检查是否包含预处理字段
        first_item = data[0] if isinstance(data, list) else data
        
        # 检查是否包含预处理特有的字段
        required_fields = ['emotion', 'text_features', 'processed_at']
        return all(field in first_item for field in required_fields)
        
    except (json.JSONDecodeError, FileNotFoundError, IndexError, KeyError):
        return False


def train_from_dialogue(dialogue_file: str, use_processed: bool = True):
    """从对话文件训练"""
    # 检查文件是否存在
    if not os.path.exists(dialogue_file):
        print(f"对话文件不存在: {dialogue_file}")
        print("请创建对话文件或使用示例文件: dialogue_examples.json")
        return
    
    # 检查对话文件是否已经预处理
    is_processed = _is_dialogue_file_processed(dialogue_file)
    
    if is_processed:
        print("=== EDASCA 快速训练模式（使用预处理数据） ===\n")
        return train_from_processed_dialogue(dialogue_file)
    else:
        print("=== EDASCA 对话训练模式（实时情感分析） ===\n")
        
        # 验证配置
        if not Config.validate():
            print("请检查.env文件中的配置")
            return
        
        # 创建EDASCA实例
        print("初始化EDASCA系统...")
        edasca = EDASCA(mode="llm")
        
        # 创建训练器
        trainer = DialogueTrainer()
        
        # 询问是否要预处理
        print("检测到未预处理的对话数据")
        choice = input("是否要先进行数据预处理以加速后续训练？(y/n): ").lower().strip()
        if choice == 'y':
            print("\n正在运行数据预处理...")
            from data_process import DataProcessor
            processor = DataProcessor()
            processor.process_dialogue_file()
            print("\n预处理完成！使用预处理数据训练...")
            return train_from_processed_dialogue("dialogue_processed.json")
        
        try:
            # 启动系统
            print("启动系统...")
            edasca.start()
            time.sleep(2)
            
            # 检查文件是否存在
            if not os.path.exists(dialogue_file):
                print(f"对话文件不存在: {dialogue_file}")
                print("请创建对话文件或使用示例文件: dialogue_examples.json")
                return
            
            # 开始训练
            print(f"\n开始从文件训练: {dialogue_file}")
            report = trainer.train_from_file(dialogue_file, edasca, save_interval=1)
            
            # 显示报告
            print("\n" + "="*50)
            print(report)
            print("="*50)
            
        except KeyboardInterrupt:
            print("\n\n训练被用户中断")
        except Exception as e:
            print(f"\n训练过程中出错: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # 停止系统
            print("\n停止EDASCA系统...")
            edasca.stop()
            print("训练完成")


def run_conversation_example(edasca):
    """运行对话示例"""
    print("\n=== 对话示例 ===")
    print("输入'quit'退出对话\n")
    
    while True:
        try:
            # 获取用户输入
            user_input = input("你: ")
            
            if user_input.lower() in ['quit', 'exit', '退出']:
                break
            
            # 处理输入
            edasca.process_input(user_input)
            time.sleep(0.5)
            
            # 获取系统状态
            status = edasca.get_status()
            
            # 简单的回复生成（基于最活跃的概念）
            implicit_tokens = status['pools']['implicit']['top_tokens']
            if implicit_tokens:
                # 这里可以集成更复杂的回复生成逻辑
                print(f"小霖: 我理解了你的意思（当前有{len(implicit_tokens)}个活跃概念）")
            
        except KeyboardInterrupt:
            break


def train_from_scenarios():
    """从场景训练（原有功能）"""
    print("=== EDASCA 场景训练模式 ===\n")
    
    # 创建EDASCA实例
    edasca = EDASCA(mode="local")
    
    # 创建训练器
    from edasca.core.data_structures import Origin
    trainer = EDASCATrainer(edasca)
    
    try:
        # 启动系统
        print("启动系统...")
        edasca.start()
        time.sleep(2)
        
        # 加载训练场景
        scenarios_file = "training_scenarios.json"
        if os.path.exists(scenarios_file):
            scenarios = trainer.load_training_scenarios(scenarios_file)
            print(f"加载了 {len(scenarios)} 个训练场景")
        else:
            print("未找到训练场景文件")
            return
        
        # 运行训练
        print("\n开始训练...")
        results = []
        
        for i, scenario in enumerate(scenarios, 1):
            print(f"\n进度: {i}/{len(scenarios)}")
            result = trainer.run_scenario(scenario)
            results.append(result)
            
            # 场景间休息
            if i < len(scenarios):
                time.sleep(2)
        
        # 生成报告
        report = trainer.generate_training_report(results)
        print(report)
        
        # 保存报告
        report_file = f"training_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"\n训练报告已保存到: {report_file}")
        
    except Exception as e:
        print(f"\n训练过程中出错: {e}")
    finally:
        edasca.stop()


class EDASCATrainer:
    """EDASCA训练器（保持向后兼容）"""
    
    def __init__(self, edasca):
        self.edasca = edasca
        self.training_data = []
        self.session_log = []
    
    def load_training_scenarios(self, filepath: str) -> List[Dict]:
        """加载训练场景"""
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def run_scenario(self, scenario: Dict) -> Dict[str, Any]:
        """运行单个训练场景"""
        print(f"\n=== 训练场景: {scenario['name']} ===")
        print(f"目标: {scenario['objective']}")
        
        results = {
            'scenario': scenario['name'],
            'steps': [],
            'start_time': datetime.now(),
            'success': False
        }
        
        try:
            # 执行场景步骤
            for step in scenario['steps']:
                step_result = self._execute_step(step)
                results['steps'].append(step_result)
                
                # 等待系统处理
                time.sleep(1)
            
            results['success'] = True
            results['end_time'] = datetime.now()
            
        except Exception as e:
            print(f"场景执行失败: {e}")
            results['error'] = str(e)
        
        return results
    
    def _execute_step(self, step: Dict) -> Dict[str, Any]:
        """执行单个步骤"""
        step_type = step['type']
        
        if step_type == 'input':
            # 处理用户输入
            print(f"用户输入: {step['content']}")
            self.edasca.process_input(step['content'])
            
            # 获取系统状态
            status = self.edasca.get_status()
            
            return {
                'type': 'input',
                'content': step['content'],
                'emotion_after': status['emotion']['emotion_label'],
                'node_count': status['database']['node_count']
            }
            
        elif step_type == 'action':
            # 执行行动
            print(f"执行行动: {step['action']}")
            
            if step['action'] == 'organize_thoughts':
                result = self.edasca.organize_thoughts()
            elif step['action'] == 'recall':
                result = self.edasca.recall_memory(step.get('query', ''))
            elif step['action'] == 'feel_state':
                result = self.edasca.feel_state()
            else:
                result = self.edasca.execute_action(step['action'])
            
            return {
                'type': 'action',
                'action': step['action'],
                'result': str(result)[:100]  # 截断长结果
            }
        
        elif step_type == 'wait':
            # 等待
            print(f"等待 {step['duration']} 秒...")
            time.sleep(step['duration'])
            
            return {
                'type': 'wait',
                'duration': step['duration']
            }
        
        elif step_type == 'expect':
            # 期望检查
            time.sleep(0.5)  # 等待系统响应
            status = self.edasca.get_status()
            
            # 检查期望
            expected = step.get('expected', {})
            checks = {}
            
            for key, expected_value in expected.items():
                if key == 'emotion_label':
                    actual = status['emotion']['emotion_label']
                    checks[key] = {
                        'expected': expected_value,
                        'actual': actual,
                        'match': actual == expected_value
                    }
                elif key.startswith('pool_size_'):
                    pool_name = key.split('_')[-1]
                    actual = status['pools'][pool_name]['size']
                    checks[key] = {
                        'expected': expected_value,
                        'actual': actual,
                        'match': actual >= expected_value  # 至少达到期望大小
                    }
            
            return {
                'type': 'expect',
                'checks': checks
            }
    
    def generate_training_report(self, results: List[Dict]) -> str:
        """生成训练报告"""
        report = ["\n=== EDASCA训练报告 ==="]
        report.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 统计
        total_scenarios = len(results)
        successful_scenarios = sum(1 for r in results if r['success'])
        
        report.append(f"\n总场景数: {total_scenarios}")
        report.append(f"成功场景: {successful_scenarios}")
        report.append(f"成功率: {successful_scenarios/total_scenarios*100:.1f}%")
        
        # 详细结果
        for result in results:
            status = "✓" if result['success'] else "✗"
            report.append(f"\n{status} {result['scenario']}")
            
            if not result['success']:
                report.append(f"  错误: {result.get('error', '未知错误')}")
            
            # 步骤摘要
            for i, step in enumerate(result['steps'], 1):
                if step['type'] == 'input':
                    report.append(f"  {i}. 输入: {step['content'][:30]}...")
                elif step['type'] == 'action':
                    report.append(f"  {i}. 行动: {step['action']}")
                elif step['type'] == 'expect':
                    matches = sum(1 for c in step['checks'].values() if c['match'])
                    total = len(step['checks'])
                    report.append(f"  {i}. 期望检查: {matches}/{total} 通过")
        
        return "\n".join(report)


def train_from_processed_dialogue(processed_file):
    """使用预处理的数据进行训练"""
    print("\n=== EDASCA 快速训练模式（使用预处理数据） ===\n")
    
    # 验证配置
    if not Config.validate():
        print("请检查.env文件中的配置")
        return
    
    # 检查预处理文件是否存在
    if not os.path.exists(processed_file):
        print(f"未找到预处理文件: {processed_file}")
        print("请先运行 python data_process.py 进行数据预处理")
        return
    
    # 创建EDASCA实例
    print("初始化EDASCA系统...")
    edasca = EDASCA(mode="llm")
    
    # 创建快速训练器
    from data_process import create_fast_trainer
    trainer = create_fast_trainer()(processed_file)
    
    try:
        # 启动系统
        print("启动系统...")
        edasca.start()
        time.sleep(2)
        
        # 开始训练
        print(f"\n开始使用预处理数据训练...")
        report = trainer.train_from_file(edasca, save_interval=1)
        
        # 显示报告
        print("\n" + "="*50)
        print(report)
        print("="*50)
        
    except KeyboardInterrupt:
        print("\n\n训练被用户中断")
    except Exception as e:
        print(f"\n训练过程中出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 停止系统
        print("\n停止EDASCA系统...")
        edasca.stop()
        print("训练完成")


def main():
    """主函数"""
    print("EDASCA 训练系统")
    print("="*50)
    print("自动检测并选择最优训练方式...\n")
    
    # 直接使用对话训练模式，会自动检测预处理文件
    dialogue_file = "dialogue.json"
    
    # 检查是否存在模型文件
    model_path = "model/model.pkl"
    if os.path.exists(model_path):
        print(f"发现现有模型: {model_path}")
        print("将从现有模型继续训练...")
    else:
        print("未发现现有模型，将从头开始训练...")
    
    # 开始训练（会自动检测文件内容是否已预处理）
    train_from_dialogue(dialogue_file)


if __name__ == "__main__":
    main()