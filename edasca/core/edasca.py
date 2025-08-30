"""
EDASCA主类
"""

import numpy as np
from typing import Dict, List, Optional, Any, Callable
import logging
import os

from .config import Config
from .data_structures import Origin, EmotionPAD
from .async_architecture import AsyncArchitecture
from ..emotion.emotion_system import EmotionLabel


class EDASCA:
    """情绪驱动激活扩散认知架构主类"""
    
    def __init__(self, mode: str = "local", **kwargs):
        """
        初始化EDASCA系统
        
        Args:
            mode: 运行模式，"llm" 或 "local"
            **kwargs: 配置参数
        """
        self.mode = mode
        self.config = Config()
        
        # 更新配置
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        
        # 初始化异步架构
        self.architecture = AsyncArchitecture()
        
        # 回调函数
        self.thought_callback: Optional[Callable] = None
        self.response_callback: Optional[Callable] = None
        
        # 设置日志
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("EDASCA")
        
        # LLM相关
        self.llm_client = None
        if mode == "llm":
            self._init_llm_client()
    
    def _init_llm_client(self):
        """初始化LLM客户端"""
        try:
            import openai
            from ..utils.env_config import Config
            
            self.llm_client = openai.OpenAI(
                api_key=Config.OPENAI_API_KEY,
                base_url=Config.OPENAI_BASE_URL
            )
            # 更新模型名称配置
            self.config.LLM_MODEL_NAME = Config.OPENAI_MODEL
        except ImportError:
            self.logger.warning("OpenAI库未安装，LLM模式将不可用")
            self.mode = "local"
    
    def start(self):
        """启动EDASCA系统"""
        self.architecture.start()
        self.logger.info(f"EDASCA系统已启动，模式: {self.mode}")
    
    def stop(self):
        """停止EDASCA系统"""
        self.architecture.stop()
        self.logger.info("EDASCA系统已停止")
    
    def process_input(self, text: str, **kwargs):
        """
        处理输入
        
        Args:
            text: 输入文本
            **kwargs: 额外参数
                - origin: 输入来源 (EXTERNAL/INTERNAL)
                - metadata: 元数据
        """
        origin = kwargs.get("origin", Origin.EXTERNAL)
        self.architecture.process_input(text, origin)
        
        # 如果是LLM模式，生成回复
        if self.mode == "llm" and origin == Origin.EXTERNAL:
            self._generate_llm_response(text)
    
    def _generate_llm_response(self, user_input: str):
        """生成LLM回复"""
        if not self.llm_client:
            return
        
        try:
            # 获取当前上下文
            context = self._get_llm_context()
            
            # 构建提示
            prompt = self._build_llm_prompt(user_input, context)
            
            # 调用LLM
            response = self.llm_client.chat.completions.create(
                model=self.config.LLM_MODEL_NAME,
                messages=[
                    {"role": "system", "content": "你是一个具有情绪和认知能力的AI助手。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.config.LLM_TEMPERATURE
            )
            
            reply = response.choices[0].message.content
            
            # 处理回复
            self._process_llm_response(reply, user_input)
            
        except Exception as e:
            self.logger.error(f"LLM生成回复失败: {e}")
    
    def _get_llm_context(self) -> Dict[str, Any]:
        """获取LLM上下文"""
        status = self.architecture.get_system_status()
        
        # 构建上下文信息
        context = {
            "current_emotion": status["emotion"],
            "active_thoughts": status["pools"]["implicit"]["top_tokens"][:5],
            "attention_focus": status["pools"]["attention"]["strongest"][:3],
            "recent_actions": status["actions"]["recent_actions"][-3:]
        }
        
        return context
    
    def _build_llm_prompt(self, user_input: str, context: Dict) -> str:
        """构建LLM提示"""
        emotion_label = context["current_emotion"]["emotion_label"]
        pad_values = context["current_emotion"]["pad"]
        
        prompt = f"""
用户输入: {user_input}

当前状态:
- 情绪: {emotion_label} (P={pad_values[0]:.2f}, A={pad_values[1]:.2f}, D={pad_values[2]:.2f})
- 活跃想法: {', '.join([t[0] for t in context['active_thoughts']])}
- 注意焦点: {', '.join([t[0] for t in context['attention_focus']])}

请基于当前状态生成合适的回复。回复应该:
1. 符合当前情绪状态
2. 考虑活跃的想法和注意焦点
3. 自然流畅

回复内容:
"""
        return prompt
    
    def _process_llm_response(self, reply: str, user_input: str):
        """处理LLM回复"""
        # 将回复作为内部输入处理
        self.architecture.process_input(reply, Origin.INTERNAL)
        
        # 调用回调
        if self.response_callback:
            self.response_callback(reply)
        
        # 学习LLM的决策
        self._learn_from_llm(user_input, reply)
    
    def _learn_from_llm(self, user_input: str, reply: str):
        """从LLM学习"""
        # 模拟LLM提供的情绪变化
        # 这里可以根据实际需要实现更复杂的学习逻辑
        delta_pad = np.array([0.1, 0.0, 0.1])  # 简单的正向情绪变化
        
        # 更新情绪系统
        self.architecture.emotion_system.apply_emotional_impact(delta_pad)
    
    def set_thought_callback(self, callback: Callable[[str], None]):
        """设置思维回调函数"""
        self.thought_callback = callback
        self.architecture.set_thought_callback(callback)
    
    def set_response_callback(self, callback: Callable[[str], None]):
        """设置回复回调函数"""
        self.response_callback = callback
    
    def execute_action(self, action_name: str, action_type: str = "internal", *args, **kwargs):
        """执行行动"""
        if action_type == "internal":
            return self.architecture.action_system.trigger_internal_action(action_name, *args, **kwargs)
        else:
            return self.architecture.action_system.trigger_external_action(action_name, *args, **kwargs)
    
    def get_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        return self.architecture.get_system_status()
    
    def get_emotion_state(self) -> Dict[str, Any]:
        """获取情绪状态"""
        return self.architecture.emotion_system.get_current_state()
    
    @property
    def emotion_system(self):
        """获取情绪系统实例"""
        return self.architecture.emotion_system
    
    @property
    def graph_db(self):
        """获取图数据库实例"""
        return self.architecture.graph_db
    
    @property
    def pool_manager(self):
        """获取激活池管理器实例"""
        return self.architecture.pool_manager
    
    def recall_memory(self, query: str) -> List[str]:
        """回忆记忆"""
        return self.execute_action("recall", "internal", query)
    
    def focus_outward(self):
        """转向外部注意"""
        return self.execute_action("focus_outward", "internal")
    
    def focus_inward(self):
        """转向内部注意"""
        return self.execute_action("focus_inward", "internal")
    
    def organize_thoughts(self) -> Dict[str, Any]:
        """整理思绪"""
        return self.execute_action("organize_thoughts", "internal")
    
    def feel_state(self) -> Dict[str, Any]:
        """感受状态"""
        return self.execute_action("feel_state", "internal")
    
    def save_state(self, filepath: str):
        """保存系统状态"""
        import pickle
        import os
        
        # 准备保存的数据 - 只保存可序列化的数据
        state_data = {
            # 保存配置的副本，避免引用不可序列化对象
            'config_data': self.config.__dict__.copy() if hasattr(self, 'config') else {},
            'mode': self.mode,
            # 训练相关信息
            'training_round': getattr(self, 'training_round', 0),
            'dialogue_index': getattr(self, 'dialogue_index', 0),
            'total_success_count': getattr(self, 'total_success_count', 0)
        }
        
        # 确保目录存在
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # 保存到文件
        with open(filepath, 'wb') as f:
            pickle.dump(state_data, f)
        
        if hasattr(self, 'logger'):
            self.logger.info(f"状态已保存到: {filepath} (轮数: {state_data['training_round']})")
    
    def load_state(self, filepath: str):
        """加载系统状态"""
        import pickle
        
        if not os.path.exists(filepath):
            if hasattr(self, 'logger'):
                self.logger.warning(f"状态文件不存在: {filepath}")
            return
        
        try:
            # 从文件加载
            with open(filepath, 'rb') as f:
                state_data = pickle.load(f)
            
            # 恢复配置
            if hasattr(self, 'config') and 'config_data' in state_data:
                # 更新现有配置对象的属性
                for key, value in state_data['config_data'].items():
                    setattr(self.config, key, value)
            
            # 恢复模式
            self.mode = state_data.get('mode', self.mode)
            
            # 恢复训练相关信息
            self.training_round = state_data.get('training_round', 0)
            self.dialogue_index = state_data.get('dialogue_index', 0)
            self.total_success_count = state_data.get('total_success_count', 0)
            
            if hasattr(self, 'logger'):
                self.logger.info(f"已从文件加载状态: {filepath} (轮数: {self.training_round})")
            
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.error(f"加载状态失败: {e}")
    
    def get_training_info(self) -> Dict[str, Any]:
        """获取训练信息"""
        return {
            'training_round': getattr(self, 'training_round', 0),
            'dialogue_index': getattr(self, 'dialogue_index', 0),
            'total_success_count': getattr(self, 'total_success_count', 0)
        }
    
    def __enter__(self):
        """上下文管理器进入"""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器退出"""
        self.stop()