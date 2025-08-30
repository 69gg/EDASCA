"""
EDASCA 对话示例
基于情绪驱动激活扩散认知架构的对话系统实现
"""

import sys
import os
import time
import numpy as np

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from edasca import EDASCA
from edasca.core.data_structures import Origin


def generate_response(edasca, user_input, status):
    """基于EDASCA架构生成回复"""
    
    # 1. 获取系统当前状态
    implicit_pool = status['pools']['implicit']
    attention_pool = status['pools']['attention']
    explicit_pool = status['pools']['explicit']
    
    # 获取情绪状态
    emotion_state = status['emotion']
    current_pad = emotion_state['pad']
    emotion_label = emotion_state['emotion_label']
    
    # 2. 基于激活扩散的想法流生成回复
    # 在本地模式下，使用隐性激活池中的内容构建回复
    
    # 2.1 获取高权重的激活概念
    activated_concepts = []
    if implicit_pool['top_tokens']:
        # 根据权重排序，选择权重最高的概念
        sorted_tokens = sorted(implicit_pool['top_tokens'], 
                             key=lambda x: x[1], reverse=True)
        
        # 选择与当前输入相关的概念
        for token, weight in sorted_tokens[:10]:
            # 检查是否与用户输入相关
            if any(word in token.lower() for word in user_input.lower().split()):
                activated_concepts.append((token, weight))
            elif len(activated_concepts) < 3:  # 即使不相关也保留一些高权重概念
                activated_concepts.append((token, weight))
    
    # 2.2 获取注意焦点
    attention_focus = []
    if attention_pool['strongest']:
        attention_focus = [token[0] for token in attention_pool['strongest'][:3]]
    
    # 3. 基于PAD情绪状态调整回复风格
    response_parts = []
    
    # 3.1 基于P值(愉悦度)选择基调
    if current_pad[0] > 0.3:
        response_parts.append("很高兴和你聊天")
    elif current_pad[0] < -0.3:
        response_parts.append("我理解你的感受")
    else:
        response_parts.append("我在听你说")
    
    # 3.2 基于激活概念生成具体内容
    if activated_concepts:
        # 使用最相关的概念
        main_concept = activated_concepts[0][0]
        
        # 根据概念生成回复
        if "你好" in user_input or "hi" in user_input.lower():
            response_parts.append(f"我是小霖，一个基于EDASCA架构的AI。当前我的情绪状态是{emotion_label}。")
        elif "天气" in user_input:
            response_parts.append(f"关于天气，我联想到{main_concept}。")
        elif "感觉" in user_input or "情绪" in user_input:
            response_parts.append(f"我现在的情绪是{emotion_label}，PAD值分别是P={current_pad[0]:.2f}, A={current_pad[1]:.2f}, D={current_pad[2]:.2f}")
        elif "为什么" in user_input:
            # 尝试推理回答
            response_parts.append(f"让我想想...根据我的理解，这可能与{main_concept}有关。")
        else:
            # 默认基于激活概念回复
            response_parts.append(f"你说的让我想到了{main_concept}。")
    
    # 3.3 结合注意焦点
    if attention_focus and len(activated_concepts) > 1:
        if attention_focus[0] not in [c[0] for c in activated_concepts]:
            response_parts.append(f"另外，我也在关注{attention_focus[0]}。")
    
    # 3.4 基于A值(唤醒度)调整回复长度和热情度
    if current_pad[1] > 0.5:
        # 高唤醒度，更热情
        response_parts.append("我对这个话题很感兴趣！")
    elif current_pad[1] < -0.2:
        # 低唤醒度，更平静
        response_parts.append("让我们慢慢聊。")
    
    # 3.5 基于D值(支配度)调整主动性
    if current_pad[2] > 0.3:
        # 高支配度，更主动
        response_parts.append("你觉得呢？")
    else:
        # 低支配度，更跟随
        response_parts.append("请继续告诉我更多。")
    
    # 4. 整合回复
    response = " ".join(response_parts)
    
    # 5. 执行内在行动以丰富对话
    try:
        # 根据对话内容触发相应的内在行动
        if "记得" in user_input or "回忆" in user_input:
            # 触发回忆
            memories = edasca.recall_memory(user_input)
            if memories:
                response += f" 我回忆起了一些相关内容：{memories[0][:50]}..."
        elif "想想" in user_input or "思考" in user_input:
            # 触发深思
            edasca.focus_inward()
            response += " 让我深入思考一下这个问题。"
        elif "感觉" in user_input:
            # 触发感受状态
            feeling = edasca.feel_state()
            response += f" 我现在的内在感受是{feeling.get('emotion_label', '平静')}。"
    except Exception as e:
        # 如果内在行动失败，不影响主要回复
        pass
    
    return response


def main():
    """主对话函数"""
    print("=== EDASCA 对话系统 ===")
    print("基于情绪驱动激活扩散认知架构的AI助手\n")
    
    # 查找模型文件
    model_dir = "model"
    model_files = []
    
    if os.path.exists(model_dir):
        for f in os.listdir(model_dir):
            # 检查两种格式的模型文件
            if (f.startswith("edasca_model_") and f.endswith(".pkl")) or f == "model.pkl":
                model_files.append(f)
    
    if not model_files:
        print("未找到训练好的模型文件")
        print("请先运行: python train.py")
        print("\n提示：")
        print("1. 准备对话数据文件")
        print("2. 运行 python data_process.py 预处理数据")
        print("3. 运行 python train.py 训练模型")
        return
    
    # 选择最新的模型
    model_files.sort(reverse=True)
    latest_model = model_files[0]
    model_path = os.path.join(model_dir, latest_model)
    
    print(f"加载模型: {latest_model}")
    
    # 创建EDASCA实例 - 使用本地模式
    edasca = EDASCA(mode="local")
    
    # 对话历史
    conversation_history = []
    max_history = 10
    
    # 设置思维回调，显示内在想法流
    def thought_callback(thought):
        print(f"[想法] {thought}")
    
    edasca.set_thought_callback(thought_callback)
    
    try:
        # 启动系统
        print("启动系统...")
        edasca.start()
        
        # 加载模型
        print("加载训练状态...")
        edasca.load_state(model_path)
        
        # 显示训练信息
        training_info = edasca.get_training_info()
        if training_info['training_round'] > 0:
            print(f"模型训练轮数: {training_info['training_round']}")
            print(f"成功对话次数: {training_info['total_success_count']}")
        
        print("\n系统就绪！输入'quit'退出对话")
        print("输入'status'查看当前状态")
        print("输入'history'查看对话历史")
        print("输入'memory'查看记忆统计")
        print("输入'thoughts'查看当前想法\n")
        
        # 对话循环
        while True:
            try:
                # 获取用户输入
                user_input = input("你: ").strip()
                
                # 处理特殊命令
                if user_input.lower() in ['quit', 'exit', '退出', '再见']:
                    print("小霖: 再见！期待下次聊天。")
                    break
                elif user_input.lower() == 'status':
                    # 显示详细状态
                    status = edasca.get_status()
                    print("\n=== 系统状态 ===")
                    
                    # 情绪状态
                    emotion = status['emotion']
                    print(f"情绪状态: {emotion['emotion_label']}")
                    print(f"PAD值: P={emotion['pad'][0]:.2f}, "
                          f"A={emotion['pad'][1]:.2f}, "
                          f"D={emotion['pad'][2]:.2f}")
                    print(f"正确感: {emotion['correctness']:.2f}")
                    print(f"违和感: {emotion['mismatch']:.2f}")
                    
                    # 激活池状态
                    print(f"\n显性激活池: {len(status['pools']['explicit']['tokens'])} 个词元")
                    print(f"隐性激活池: {len(status['pools']['implicit']['top_tokens'])} 个活跃概念")
                    print(f"注意激活池: {len(status['pools']['attention']['strongest'])} 个焦点")
                    print(f"行动激活池: {len(status['pools']['action']['active_actions'])} 个待执行行动")
                    
                    # 记忆状态
                    memory = status['memory']
                    print(f"情景记忆: {len(memory['episodic'])} 条")
                    print(f"语义记忆: {memory['concept_count']} 个概念")
                    
                    print("=================\n")
                    continue
                elif user_input.lower() == 'history':
                    # 显示对话历史
                    print("\n=== 对话历史 ===")
                    for i, (user_msg, bot_msg) in enumerate(conversation_history[-5:], 1):
                        print(f"{i}. 你: {user_msg}")
                        print(f"   小霖: {bot_msg}\n")
                    print("=================\n")
                    continue
                elif user_input.lower() == 'memory':
                    # 显示记忆统计
                    try:
                        # 触发整理思绪行动
                        thoughts = edasca.organize_thoughts()
                        print("\n=== 记忆与思绪 ===")
                        print(f"当前思绪: {thoughts}")
                        
                        # 显示激活的概念
                        status = edasca.get_status()
                        implicit_tokens = status['pools']['implicit']['top_tokens'][:5]
                        print(f"活跃概念: {', '.join([t[0] for t in implicit_tokens])}")
                        
                        # 显示注意焦点
                        attention_tokens = status['pools']['attention']['strongest'][:3]
                        print(f"注意焦点: {', '.join([t[0] for t in attention_tokens])}")
                        
                        print("=================\n")
                    except Exception as e:
                        print(f"获取记忆信息失败: {e}\n")
                    continue
                elif user_input.lower() == 'thoughts':
                    # 显示当前想法流
                    print("\n=== 当前想法流 ===")
                    print("系统正在生成内在想法...")
                    # 触发一次内向思考
                    edasca.focus_inward()
                    time.sleep(1)  # 给系统一些时间生成想法
                    print("=================\n")
                    continue
                elif not user_input:
                    print("小霖: 请输入一些内容吧。")
                    continue
                
                # 处理输入
                print("\n[处理输入...]")
                edasca.process_input(user_input)
                
                # 给系统一些时间进行激活扩散和想法生成
                time.sleep(0.5)
                
                # 获取系统状态
                status = edasca.get_status()
                
                # 生成回复
                response = generate_response(edasca, user_input, status)
                
                # 输出回复
                print(f"\n小霖: {response}\n")
                
                # 记录对话历史
                conversation_history.append((user_input, response))
                if len(conversation_history) > max_history:
                    conversation_history.pop(0)
                
                # 显示简要状态信息
                emotion = status['emotion']['emotion_label']
                active_count = len(status['pools']['implicit']['top_tokens'])
                print(f"[状态: {emotion}, 活跃概念: {active_count}]")
                
            except KeyboardInterrupt:
                print("\n\n对话已结束")
                break
            except Exception as e:
                print(f"\n错误: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    except Exception as e:
        print(f"系统错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 停止系统
        print("\n停止系统...")
        edasca.stop()


if __name__ == "__main__":
    main()