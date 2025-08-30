"""
核心数据结构定义
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import uuid
import numpy as np


class Origin(Enum):
    """起源类型"""
    EXTERNAL = "external"  # 外部来源
    INTERNAL = "internal"  # 内部来源


class NodeType(Enum):
    """节点类型"""
    WORD = "word"          # 普通词汇
    SPECIAL = "special"    # 特殊节点（如行动标识符）


@dataclass
class ConceptNode:
    """概念节点"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    type: NodeType = NodeType.WORD
    origin: Origin = Origin.EXTERNAL
    base_weight: float = 0.1
    last_activated: datetime = field(default_factory=datetime.now)
    emotion_ema: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))  # P, A, D
    
    def update_base_weight(self, activation_reward: float = None):
        """更新基础权重"""
        if activation_reward is None:
            activation_reward = 0.01
        
        self.base_weight = (self.base_weight * 0.995) + (activation_reward * 0.005)
        self.last_activated = datetime.now()
    
    def update_emotion_ema(self, current_emotion: np.ndarray, alpha: float = 0.3):
        """更新情绪EMA"""
        self.emotion_ema = (1 - alpha) * self.emotion_ema + alpha * current_emotion


@dataclass
class RelationEdge:
    """关系边"""
    source_id: str
    target_id: str
    frequency: int = 1
    avg_time_delta: float = 0.0
    recent_emotion_delta_ema: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))
    ema_alpha: float = 0.5
    last_updated: datetime = field(default_factory=datetime.now)
    
    def update_emotion_ema(self, new_delta: np.ndarray):
        """更新情绪变化EMA"""
        self.recent_emotion_delta_ema = (
            (1 - self.ema_alpha) * self.recent_emotion_delta_ema + 
            self.ema_alpha * new_delta
        )
        self.last_updated = datetime.now()
    
    def update_frequency(self):
        """更新共现频次"""
        self.frequency += 1
        self.last_updated = datetime.now()


@dataclass
class SensoryMemoryNode:
    """感受记忆节点"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    origin: Origin = Origin.EXTERNAL
    timestamp: datetime = field(default_factory=datetime.now)
    emotion_at_encoding: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))
    importance: float = 0.5
    links: List[str] = field(default_factory=list)  # 连接的认知节点ID列表
    
    def update_importance(self, activation_strength: float):
        """更新重要性权重"""
        self.importance = self.importance * 0.9 + activation_strength * 0.1


@dataclass
class ActivationToken:
    """激活令牌"""
    node_id: str
    content: str
    weight: float
    origin: Origin
    emotion_snapshot: np.ndarray
    decay_rate: float = 0.1
    created_at: datetime = field(default_factory=datetime.now)
    
    def decay(self, dt: float):
        """衰减权重"""
        self.weight *= np.exp(-self.decay_rate * dt)


@dataclass
class AttentionToken:
    """注意令牌"""
    node_id: str
    expectation_value: float = 0.0  # 期待值（正值）
    pressure_value: float = 0.0     # 压力值（负值）
    precursor_nodes: List[str] = field(default_factory=list)
    time_expectation_factor: float = 1.0
    created_at: datetime = field(default_factory=datetime.now)
    
    @property
    def absolute_value(self) -> float:
        """获取绝对值"""
        return abs(self.expectation_value) if self.expectation_value != 0 else abs(self.pressure_value)
    
    def decay(self, dt: float):
        """衰减"""
        decay_rate = 0.01 * self.time_expectation_factor
        if self.expectation_value != 0:
            self.expectation_value *= np.exp(-decay_rate * dt)
        else:
            self.pressure_value *= np.exp(-decay_rate * dt)


@dataclass
class ActionToken:
    """行动令牌"""
    node_id: str
    weight: float
    action_type: str
    created_at: datetime = field(default_factory=datetime.now)
    
    def update_weight(self, delta: float):
        """更新权重"""
        self.weight += delta
        self.weight = max(0.0, min(1.0, self.weight))  # 限制在[0,1]范围内


class EmotionPAD:
    """PAD情绪状态"""
    
    def __init__(self, p: float = 0.0, a: float = 0.0, d: float = 0.0):
        self.p = p  # Pleasure 愉悦度
        self.a = a  # Arousal 唤醒度
        self.d = d  # Dominance 支配度
    
    def to_array(self) -> np.ndarray:
        """转换为numpy数组"""
        return np.array([self.p, self.a, self.d])
    
    def from_array(self, arr: np.ndarray):
        """从numpy数组加载"""
        self.p, self.a, self.d = arr
    
    def update(self, delta_p: float, delta_a: float, delta_d: float):
        """更新情绪值"""
        self.p += delta_p
        self.a += delta_a
        self.d += delta_d
        
        # 限制在[-1,1]范围内
        self.p = max(-1.0, min(1.0, self.p))
        self.a = max(-1.0, min(1.0, self.a))
        self.d = max(-1.0, min(1.0, self.d))
    
    def regress_to_baseline(self, baseline: Tuple[float, float, float], mu: float = 0.05):
        """回归到基线"""
        for i, (current, base) in enumerate(zip([self.p, self.a, self.d], baseline)):
            regression = (base - current) * mu * (1 + abs(current - base) ** 2)
            if i == 0:
                self.p += regression
            elif i == 1:
                self.a += regression
            else:
                self.d += regression