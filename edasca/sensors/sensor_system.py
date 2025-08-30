"""
感受器系统
"""

import numpy as np
import re
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
import math

from ..core.data_structures import ActivationToken, SensoryMemoryNode, Origin
from ..core.config import Config
from ..emotion.emotion_system import EmotionSystem
from ..pools.pool_manager import PoolManager
from ..database.graph_db import GraphDatabase


class TimeSensor:
    """时间感受器"""
    
    def __init__(self, graph_db: GraphDatabase, pool_manager: PoolManager):
        self.graph_db = graph_db
        self.pool_manager = pool_manager
        self.time_keywords = self._build_time_keywords()
    
    def _build_time_keywords(self) -> Dict[str, float]:
        """构建时间关键词词典"""
        return {
            # 时间单位
            "秒": 1.0,
            "分钟": 60.0,
            "小时": 3600.0,
            "天": 86400.0,
            "周": 604800.0,
            "月": 2592000.0,
            "年": 31536000.0,
            
            # 时间描述
            "刚才": 10.0,
            "刚刚": 30.0,
            "不久前": 300.0,
            "今天": 0.0,
            "昨天": 86400.0,
            "前天": 172800.0,
            "明天": -86400.0,
            "后天": -172800.0,
            
            # 相对时间
            "早": -3600.0,
            "晚": 3600.0,
            "上": -86400.0,
            "下": 86400.0,
        }
    
    def process_memory(self, memory: SensoryMemoryNode, current_emotion: np.ndarray) -> None:
        """处理感受记忆，生成时间感受"""
        # 计算时间差
        time_diff = (datetime.now() - memory.timestamp).total_seconds()
        
        # 生成时间感受令牌
        time_feeling = self._generate_time_feeling(time_diff)
        
        if time_feeling:
            # 创建激活令牌
            token = ActivationToken(
                node_id=f"time_interval:{time_diff}",
                content=time_feeling,
                weight=0.5,
                origin=Origin.INTERNAL,
                emotion_snapshot=current_emotion.copy()
            )
            
            # 添加到显性池
            self.pool_manager.add_to_explicit(token)
            
            # 执行时间扩散激活
            self._time_spreading_activation(time_diff, current_emotion)
    
    def _generate_time_feeling(self, time_diff: float) -> Optional[str]:
        """生成时间感受描述"""
        if time_diff < 60:
            return f"刚刚"
        elif time_diff < 3600:
            minutes = int(time_diff / 60)
            return f"{minutes}分钟前"
        elif time_diff < 86400:
            hours = int(time_diff / 3600)
            return f"{hours}小时前"
        elif time_diff < 604800:
            days = int(time_diff / 86400)
            return f"{days}天前"
        elif time_diff < 2592000:
            weeks = int(time_diff / 604800)
            return f"{weeks}周前"
        elif time_diff < 31536000:
            months = int(time_diff / 2592000)
            return f"{months}个月前"
        else:
            years = int(time_diff / 31536000)
            return f"{years}年前"
    
    def _time_spreading_activation(self, time_diff: float, current_emotion: np.ndarray) -> None:
        """时间扩散激活"""
        S_base = 1.0 + current_emotion[2]  # 1 + D值
        
        # 查找所有时间相关的节点
        for content, target_time in self.time_keywords.items():
            # 计算时间相似度
            delta_t = abs(time_diff - target_time)
            f = Config.TIME_FUZZINESS_FACTOR
            similarity = max(0, 1 - (delta_t / (f * target_time))) if target_time > 0 else 0
            
            if similarity > 0.1:  # 相似度阈值
                # 计算激活增量
                delta_activation = S_base * similarity
                
                # 查找或创建时间节点
                node = self.graph_db.get_node_by_content(content, Origin.INTERNAL)
                if not node:
                    from ..core.data_structures import ConceptNode, NodeType
                    node = ConceptNode(
                        content=content,
                        type=NodeType.WORD,
                        origin=Origin.INTERNAL
                    )
                    node_id = self.graph_db.add_node(node)
                else:
                    node_id = node.id
                
                # 应用激活
                self._apply_time_activation(node_id, delta_activation, current_emotion)
    
    def _apply_time_activation(self, node_id: str, delta: float, emotion_snapshot: np.ndarray) -> None:
        """应用时间激活"""
        existing_token = self.pool_manager.implicit_pool.get_token(node_id)
        
        if existing_token:
            existing_token.weight += delta
            existing_token.emotion_snapshot = emotion_snapshot.copy()
        else:
            node = self.graph_db.get_node(node_id)
            if node:
                token = ActivationToken(
                    node_id=node_id,
                    content=node.content,
                    weight=max(0.0, node.base_weight + delta),
                    origin=Origin.INTERNAL,
                    emotion_snapshot=emotion_snapshot.copy()
                )
                self.pool_manager.add_to_implicit(token)
    
    def parse_time_expression(self, text: str) -> Optional[float]:
        """解析时间表达式"""
        # 匹配数字+时间单位
        pattern = r'(\d+)\s*([秒分钟小时天周月年])'
        match = re.search(pattern, text)
        
        if match:
            num = int(match.group(1))
            unit = match.group(2)
            return num * self.time_keywords.get(unit, 1.0)
        
        # 匹配相对时间
        for keyword, time_value in self.time_keywords.items():
            if keyword in text:
                return time_value
        
        return None


class EmotionSensor:
    """情绪感受器"""
    
    def __init__(self, graph_db: GraphDatabase, pool_manager: PoolManager, 
                 emotion_system: EmotionSystem):
        self.graph_db = graph_db
        self.pool_manager = pool_manager
        self.emotion_system = emotion_system
    
    def check_triggers(self) -> bool:
        """检查是否需要触发情绪感受器"""
        emotion_state = self.emotion_system.get_current_state()
        pad = np.array(emotion_state["pad"])
        
        # 检查各种触发条件
        triggers = []
        
        # 1. 情绪强度触发
        if any(abs(x) > Config.EMOTION_INTENSITY_THRESHOLD for x in pad):
            triggers.append("intensity")
        
        # 2. 情绪变化触发
        if hasattr(self, '_last_pad'):
            delta_pad = pad - self._last_pad
            if any(abs(x) > Config.EMOTION_CHANGE_THRESHOLD for x in delta_pad):
                triggers.append("change")
        
        # 3. 正确感/违和感触发
        if abs(emotion_state["correctness"]) > 0.5:
            triggers.append("correctness")
        
        # 4. 注意触发
        for attention in self.pool_manager.attention_pool.get_strongest_attentions():
            if attention.absolute_value > 0.5:
                triggers.append("attention")
                break
        
        # 保存当前状态
        self._last_pad = pad.copy()
        
        return len(triggers) > 0
    
    def generate_emotion_feeling(self) -> Optional[ActivationToken]:
        """生成情绪感受"""
        if not self.check_triggers():
            return None
        
        # 获取当前情绪标签
        emotion_label, similarity = self.emotion_system.get_emotion_label()
        
        if similarity > 0.5:  # 相似度阈值
            # 生成情绪感受
            content = f"emotion:{emotion_label.value}"
            
            # 查找或创建情绪节点
            node = self.graph_db.get_node_by_content(content, Origin.INTERNAL)
            if not node:
                from ..core.data_structures import ConceptNode, NodeType
                node = ConceptNode(
                    content=content,
                    type=NodeType.SPECIAL,
                    origin=Origin.INTERNAL
                )
                node_id = self.graph_db.add_node(node)
            else:
                node_id = node.id
            
            # 创建激活令牌
            token = ActivationToken(
                node_id=node_id,
                content=emotion_label.value,
                weight=similarity,
                origin=Origin.INTERNAL,
                emotion_snapshot=self.emotion_system.get_emotion_vector()
            )
            
            # 添加到显性池
            self.pool_manager.add_to_explicit(token)
            
            # 执行情绪扩散激活
            self._emotion_spreading_activation(emotion_label.value, 
                                              self.emotion_system.get_emotion_vector())
            
            return token
        
        return None
    
    def _emotion_spreading_activation(self, emotion_label: str, 
                                    current_emotion: np.ndarray) -> None:
        """情绪扩散激活"""
        S_base = 1.0 + current_emotion[2]  # 1 + D值
        
        # 查找所有与该情绪相关的节点
        emotion_nodes = []
        for node_id, node in self.graph_db.nodes.items():
            if node.origin == Origin.INTERNAL and emotion_label in node.content.lower():
                emotion_nodes.append(node)
        
        # 计算每个节点的情绪相似度
        for node in emotion_nodes:
            # 计算情绪相似度
            norm_current = np.linalg.norm(current_emotion)
            norm_node = np.linalg.norm(node.emotion_ema)
            
            # 避免除零错误
            if norm_current > 0 and norm_node > 0:
                similarity = np.dot(current_emotion, node.emotion_ema) / (norm_current * norm_node)
            else:
                similarity = 0.0
            
            if similarity > 0.1:  # 相似度阈值
                # 计算激活增量
                delta_activation = S_base * similarity
                
                # 应用激活
                existing_token = self.pool_manager.implicit_pool.get_token(node.id)
                if existing_token:
                    existing_token.weight += delta_activation
                    existing_token.emotion_snapshot = current_emotion.copy()
                else:
                    token = ActivationToken(
                        node_id=node.id,
                        content=node.content,
                        weight=max(0.0, node.base_weight + delta_activation),
                        origin=Origin.INTERNAL,
                        emotion_snapshot=current_emotion.copy()
                    )
                    self.pool_manager.add_to_implicit(token)


class SensorSystem:
    """感受器系统管理器"""
    
    def __init__(self, graph_db: GraphDatabase, pool_manager: PoolManager, 
                 emotion_system: EmotionSystem):
        self.time_sensor = TimeSensor(graph_db, pool_manager)
        self.emotion_sensor = EmotionSensor(graph_db, pool_manager, emotion_system)
    
    def process_sensory_input(self, memory: SensoryMemoryNode, current_emotion: np.ndarray) -> None:
        """处理感知输入"""
        # 时间感受器处理
        self.time_sensor.process_memory(memory, current_emotion)
        
        # 情绪感受器处理
        self.emotion_sensor.generate_emotion_feeling()
    
    def parse_time_from_text(self, text: str) -> Optional[float]:
        """从文本中解析时间"""
        return self.time_sensor.parse_time_expression(text)