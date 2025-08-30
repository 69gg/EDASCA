"""
多线程异步架构
"""

import threading
import time
import queue
import numpy as np
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime
import logging

from ..core.data_structures import Origin, SensoryMemoryNode, ActivationToken
from ..pools.pool_manager import PoolManager
from ..database.graph_db import GraphDatabase
from ..emotion.emotion_system import EmotionSystem
from ..core.tokenizer import DynamicTokenizer
from ..core.activation_spreading import ActivationSpreading
from ..sensors.sensor_system import SensorSystem
from ..actions.action_system import ActionSystem
from ..core.config import Config


class InputProcessor:
    """输入处理器"""
    
    def __init__(self, pool_manager: PoolManager, graph_db: GraphDatabase,
                 tokenizer: DynamicTokenizer, spreading: ActivationSpreading,
                 emotion_system: EmotionSystem, sensor_system: SensorSystem):
        self.pool_manager = pool_manager
        self.graph_db = graph_db
        self.tokenizer = tokenizer
        self.spreading = spreading
        self.emotion_system = emotion_system
        self.sensor_system = sensor_system
        self.input_queue = queue.Queue()
        self.running = False
        self.thread = None
    
    def start(self):
        """启动输入处理线程"""
        self.running = True
        self.thread = threading.Thread(target=self._process_loop, daemon=True)
        self.thread.start()
    
    def stop(self):
        """停止输入处理线程"""
        self.running = False
        if self.thread:
            self.thread.join()
    
    def add_input(self, text: str, origin: Origin, metadata: Dict = None):
        """添加输入"""
        self.input_queue.put({
            "text": text,
            "origin": origin,
            "metadata": metadata or {},
            "timestamp": datetime.now()
        })
    
    def _process_loop(self):
        """处理循环"""
        while self.running:
            try:
                # 获取输入
                input_data = self.input_queue.get(timeout=0.1)
                
                # 处理输入
                self._process_input(input_data)
                
            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"输入处理错误: {e}")
    
    def _process_input(self, input_data: Dict):
        """处理单个输入"""
        text = input_data["text"]
        origin = input_data["origin"]
        timestamp = input_data["timestamp"]
        
        # 1. 动态分词
        tokens = self.tokenizer.tokenize(text, origin)
        
        # 2. 添加到显性激活池
        for token in tokens:
            self.pool_manager.add_to_explicit(token)
        
        # 3. 更新正确感/违和感
        self._update_correctness(tokens)
        
        # 4. 执行激活扩散
        current_emotion = self.emotion_system.get_emotion_vector()
        node_ids = [token.node_id for token in tokens]
        self.spreading.batch_spread(node_ids, current_emotion)
        
        # 5. 更新数据库
        self._update_database(tokens, origin, current_emotion)
        
        # 6. 如果是外部输入或来自感受器，创建感受记忆
        if origin == Origin.EXTERNAL or input_data.get("from_sensor", False):
            memory = SensoryMemoryNode(
                content=text,
                origin=origin,
                timestamp=timestamp,
                emotion_at_encoding=current_emotion.copy(),
                links=node_ids
            )
            self.graph_db.add_sensory_memory(memory)
            
            # 7. 触发感受器
            self.sensor_system.process_sensory_input(memory, current_emotion)
    
    def _update_correctness(self, tokens: List[ActivationToken]):
        """更新正确感/违和感"""
        # 简单的匹配逻辑：如果词元在知识库中存在，增加正确感
        matched_count = 0
        for token in tokens:
            if self.graph_db.get_node(token.node_id):
                matched_count += 1
        
        match_ratio = matched_count / len(tokens) if tokens else 0
        self.emotion_system.update_correctness(
            matched=match_ratio > 0.5,
            strength=match_ratio * 0.1
        )
    
    def _update_database(self, tokens: List[ActivationToken], origin: Origin, 
                        emotion: np.ndarray):
        """更新数据库"""
        # 获取显性池中的概念作为前置节点
        explicit_tokens = self.pool_manager.explicit_pool.get_top_tokens(10)
        precursor_ids = [t.node_id for t in explicit_tokens if t.node_id not in [token.node_id for token in tokens]]
        
        # 建立时序边
        if precursor_ids and tokens:
            for prec_id in precursor_ids:
                self.graph_db.add_edge(prec_id, tokens[0].node_id, 
                                     emotion_delta=emotion)
        
        # 建立共现边
        for i in range(len(tokens) - 1):
            self.graph_db.add_edge(tokens[i].node_id, tokens[i+1].node_id,
                                 emotion_delta=emotion)
        
        # 更新节点的激活时间和情绪
        for token in tokens:
            node = self.graph_db.get_node(token.node_id)
            if node:
                node.update_base_weight()
                node.update_emotion_ema(emotion)


class ThoughtStreamGenerator:
    """思维流生成器"""
    
    def __init__(self, pool_manager: PoolManager, graph_db: GraphDatabase,
                 tokenizer: DynamicTokenizer, spreading: ActivationSpreading,
                 emotion_system: EmotionSystem):
        self.pool_manager = pool_manager
        self.graph_db = graph_db
        self.tokenizer = tokenizer
        self.spreading = spreading
        self.emotion_system = emotion_system
        self.running = False
        self.thread = None
        self.thought_callback: Optional[Callable] = None
    
    def start(self):
        """启动思维流线程"""
        self.running = True
        self.thread = threading.Thread(target=self._generate_loop, daemon=True)
        self.thread.start()
    
    def stop(self):
        """停止思维流线程"""
        self.running = False
        if self.thread:
            self.thread.join()
    
    def set_thought_callback(self, callback: Callable[[str], None]):
        """设置思维回调"""
        self.thought_callback = callback
    
    def _generate_loop(self):
        """生成循环"""
        while self.running:
            try:
                # 检查是否有足够的激活
                if self._should_generate_thought():
                    thought = self._generate_thought()
                    if thought:
                        # 将思维作为内部输入
                        self.pool_manager.add_to_explicit(thought)
                        
                        # 调用回调
                        if self.thought_callback:
                            self.thought_callback(thought.content)
                
                time.sleep(0.1)  # 控制生成频率
                
            except Exception as e:
                logging.error(f"思维流生成错误: {e}")
    
    def _should_generate_thought(self) -> bool:
        """判断是否应该生成思维"""
        # 简单策略：隐性池中有足够强度的激活
        top_tokens = self.pool_manager.implicit_pool.get_top_tokens(5)
        return len(top_tokens) > 0 and top_tokens[0].weight > 0.3
    
    def _generate_thought(self) -> Optional[ActivationToken]:
        """生成思维"""
        # 获取候选词元
        candidates = self.pool_manager.implicit_pool.get_top_tokens(20)
        
        if not candidates:
            return None
        
        # 选择权重最高的词元
        selected = candidates[0]
        
        # 创建思维令牌
        thought = ActivationToken(
            node_id=selected.node_id,
            content=selected.content,
            weight=selected.weight,
            origin=Origin.INTERNAL,
            emotion_snapshot=self.emotion_system.get_emotion_vector()
        )
        
        # 执行激活扩散
        self.spreading.spread_activation(
            selected.node_id,
            self.emotion_system.get_emotion_vector()
        )
        
        return thought


class ActionExecutor:
    """行动执行器"""
    
    def __init__(self, action_system: ActionSystem):
        self.action_system = action_system
        self.running = False
        self.thread = None
    
    def start(self):
        """启动行动执行线程"""
        self.running = True
        self.thread = threading.Thread(target=self._execute_loop, daemon=True)
        self.thread.start()
    
    def stop(self):
        """停止行动执行线程"""
        self.running = False
        if self.thread:
            self.thread.join()
    
    def _execute_loop(self):
        """执行循环"""
        while self.running:
            try:
                # 检查并执行准备好的行动
                results = self.action_system.execute_ready_actions()
                
                if results:
                    for result in results:
                        logging.info(f"执行行动: {result['action']}, 结果: {result['success']}")
                
                time.sleep(0.05)  # 高频检查
                
            except Exception as e:
                logging.error(f"行动执行错误: {e}")


class EmotionUpdater:
    """情绪更新器"""
    
    def __init__(self, emotion_system: EmotionSystem, pool_manager: PoolManager):
        self.emotion_system = emotion_system
        self.pool_manager = pool_manager
        self.running = False
        self.thread = None
    
    def start(self):
        """启动情绪更新线程"""
        self.running = True
        self.thread = threading.Thread(target=self._update_loop, daemon=True)
        self.thread.start()
    
    def stop(self):
        """停止情绪更新线程"""
        self.running = False
        if self.thread:
            self.thread.join()
    
    def _update_loop(self):
        """更新循环"""
        while self.running:
            try:
                # 应用情绪回归
                self.emotion_system.update_emotion()
                
                # 更新时间
                time.sleep(Config.EMOTION_UPDATE_INTERVAL)
                
            except Exception as e:
                logging.error(f"情绪更新错误: {e}")


class AsyncArchitecture:
    """异步架构管理器"""
    
    def __init__(self):
        # 初始化各个组件
        self.graph_db = GraphDatabase()
        self.pool_manager = PoolManager()
        self.emotion_system = EmotionSystem()
        self.tokenizer = DynamicTokenizer(self.pool_manager, self.graph_db)
        self.spreading = ActivationSpreading(self.pool_manager, self.graph_db)
        self.sensor_system = SensorSystem(self.graph_db, self.pool_manager, 
                                         self.emotion_system)
        self.action_system = ActionSystem(self.pool_manager, self.graph_db, 
                                         self.emotion_system)
        
        # 初始化异步组件
        self.input_processor = InputProcessor(
            self.pool_manager, self.graph_db, self.tokenizer,
            self.spreading, self.emotion_system, self.sensor_system
        )
        self.thought_stream = ThoughtStreamGenerator(
            self.pool_manager, self.graph_db, self.tokenizer,
            self.spreading, self.emotion_system
        )
        self.action_executor = ActionExecutor(self.action_system)
        self.emotion_updater = EmotionUpdater(self.emotion_system, self.pool_manager)
        
        # 系统状态
        self.running = False
    
    def start(self):
        """启动系统"""
        if self.running:
            return
        
        self.running = True
        
        # 启动各个线程
        self.input_processor.start()
        self.thought_stream.start()
        self.action_executor.start()
        self.emotion_updater.start()
        
        logging.info("EDASCA异步架构已启动")
    
    def stop(self):
        """停止系统"""
        if not self.running:
            return
        
        self.running = False
        
        # 停止各个线程
        self.input_processor.stop()
        self.thought_stream.stop()
        self.action_executor.stop()
        self.emotion_updater.stop()
        
        logging.info("EDASCA异步架构已停止")
    
    def process_input(self, text: str, origin: Origin = Origin.EXTERNAL):
        """处理输入"""
        self.input_processor.add_input(text, origin)
    
    def set_thought_callback(self, callback: Callable[[str], None]):
        """设置思维回调"""
        self.thought_stream.set_thought_callback(callback)
    
    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        return {
            "running": self.running,
            "pools": self.pool_manager.get_pool_summary(),
            "emotion": self.emotion_system.get_current_state(),
            "database": self.graph_db.get_statistics(),
            "actions": self.action_system.get_action_statistics()
        }