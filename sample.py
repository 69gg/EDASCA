"""
EDASCA 对话示例
运行训练后的模型进行对话
"""

import sys
import os
import time

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from edasca import EDASCA


def main():
    """主对话函数"""
    print("=== EDASCA 对话系统 ===\n")
    
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
        return
    
    # 选择最新的模型
    model_files.sort(reverse=True)
    latest_model = model_files[0]
    model_path = os.path.join(model_dir, latest_model)
    
    print(f"加载模型: {latest_model}\n")
    
    # 创建EDASCA实例
    edasca = EDASCA(mode="local")
    
    try:
        # 启动系统
        print("启动系统...")
        edasca.start()
        
        # 加载模型
        print("加载训练状态...")
        edasca.load_state(model_path)
        
        print("\n系统就绪！输入'quit'退出对话\n")
        
        # 对话循环
        while True:
            try:
                # 获取用户输入
                user_input = input("你: ")
                
                if user_input.lower() in ['quit', 'exit', '退出', '再见']:
                    print("小霖: 再见！期待下次聊天。")
                    break
                
                # 处理输入
                edasca.process_input(user_input)
                time.sleep(0.5)
                
                # 获取系统状态
                status = edasca.get_status()
                
                # 简单的回复生成
                implicit_tokens = status['pools']['implicit']['top_tokens']
                
                # 查找最相关的回复
                response = generate_response(edasca, user_input, status)
                
                print(f"小霖: {response}")
                
                # 显示状态信息（可选）
                if len(implicit_tokens) > 0:
                    emotion = status['emotion']['emotion_label']
                    print(f"[状态: {emotion}, 活跃概念: {len(implicit_tokens)}]")
                
            except KeyboardInterrupt:
                print("\n\n对话已结束")
                break
            except Exception as e:
                print(f"\n错误: {e}")
                continue
    
    except Exception as e:
        print(f"系统错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 停止系统
        print("\n停止系统...")
        edasca.stop()


def generate_response(edasca, user_input, status):
    """生成回复"""
    # 简单的回复生成策略
    
    # 1. 检查是否有直接匹配的训练数据
    # 这里可以通过搜索记忆库来找到最佳回复
    
    # 2. 基于激活概念生成回复
    implicit_tokens = status['pools']['implicit']['top_tokens'][:5]
    
    # 3. 简单的回复模板
    responses = {
        "问候": ["你好！", "嗨！", "很高兴见到你！"],
        "身份": ["我是小霖。", "我是小霖，一个AI助手。"],
        "创造者": ["潘子创造了我。", "我的创造者是潘子。"],
        "天气": ["希望今天天气不错！", "天气如何呢？"],
        "感谢": ["不客气！", "很高兴能帮到你！"],
        "再见": ["再见！", "期待下次见面！"]
    }
    
    # 简单的关键词匹配
    for keyword, reply_list in responses.items():
        if keyword in user_input:
            return reply_list[0]
    
    # 默认回复
    if implicit_tokens:
        # 基于激活的概念生成回复
        concepts = [token[0] for token in implicit_tokens if token[0] in user_input]
        if concepts:
            return f"我理解你在说'{concepts[0]}'。"
    
    return "我明白了，请继续说。"


if __name__ == "__main__":
    main()