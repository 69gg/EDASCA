"""
激活池管理系统
"""

from typing import Dict, List, Optional, Tuple
from datetime import datetime
import numpy as np
from collections import defaultdict
import threading
import time

from ..core.data_structures import (
    ActivationToken, AttentionToken, ActionToken, 
    Origin, ConceptNode
)
from ..core.config import Config


class ActivationPool:
    """激活池基类"""
    
    def __init__(self, max_size: int, name: str):
        self.max_size = max_size
        self.name = name
        self.tokens: Dict[str, ActivationToken] = {}
        self._lock = threading.Lock()
    
    def add_token(self, token: ActivationToken):
        """添加激活令牌"""
        with self._lock:
            if len(self.tokens) >= self.max_size:
                self._evict_weakest()
            self.tokens[token.node_id] = token
    
    def get_token(self, node_id: str) -> Optional[ActivationToken]:
        """获取激活令牌"""
        with self._lock:
            return self.tokens.get(node_id)
    
    def update_weight(self, node_id: str, delta: float):
        """更新令牌权重"""
        with self._lock:
            if node_id in self.tokens:
                self.tokens[node_id].weight += delta
                self.tokens[node_id].weight = max(0.0, self.tokens[node_id].weight)
    
    def remove_token(self, node_id: str):
        """移除激活令牌"""
        with self._lock:
            if node_id in self.tokens:
                del self.tokens[node_id]
    
    def decay_all(self, dt: float):
        """所有令牌衰减"""
        with self._lock:
            for token in self.tokens.values():
                token.decay(dt)
    
    def get_top_tokens(self, n: int = 10) -> List[ActivationToken]:
        """获取权重最高的n个令牌"""
        with self._lock:
            sorted_tokens = sorted(
                self.tokens.values(), 
                key=lambda x: x.weight, 
                reverse=True
            )
            return sorted_tokens[:n]
    
    def get_tokens_by_origin(self, origin: Origin) -> List[ActivationToken]:
        """根据起源获取令牌"""
        with self._lock:
            return [token for token in self.tokens.values() if token.origin == origin]
    
    def clear(self):
        """清空激活池"""
        with self._lock:
            self.tokens.clear()
    
    def _evict_weakest(self):
        """淘汰最弱的令牌"""
        if self.tokens:
            weakest_id = min(self.tokens.items(), key=lambda x: x[1].weight)[0]
            del self.tokens[weakest_id]
    
    def size(self) -> int:
        """获取池大小"""
        with self._lock:
            return len(self.tokens)


class ExplicitActivationPool(ActivationPool):
    """显性激活池"""
    
    def __init__(self):
        super().__init__(Config.MAX_EXPLICIT_POOL_SIZE, "explicit")


class ImplicitActivationPool(ActivationPool):
    """隐性激活池"""
    
    def __init__(self):
        super().__init__(Config.MAX_IMPLICIT_POOL_SIZE, "implicit")


class AttentionActivationPool:
    """注意激活池"""
    
    def __init__(self):
        self.max_size = Config.MAX_ATTENTION_POOL_SIZE
        self.tokens: Dict[str, AttentionToken] = {}
        self._lock = threading.Lock()
    
    def add_attention(self, node_id: str, expectation: float = None, pressure: float = None, 
                     precursors: List[str] = None):
        """添加注意令牌"""
        with self._lock:
            if len(self.tokens) >= self.max_size:
                self._evict_weakest()
            
            token = AttentionToken(
                node_id=node_id,
                expectation_value=expectation or 0.0,
                pressure_value=pressure or 0.0,
                precursor_nodes=precursors or []
            )
            self.tokens[node_id] = token
    
    def get_attention(self, node_id: str) -> Optional[AttentionToken]:
        """获取注意令牌"""
        with self._lock:
            return self.tokens.get(node_id)
    
    def update_attention(self, node_id: str, delta_expectation: float = None, 
                        delta_pressure: float = None):
        """更新注意值"""
        with self._lock:
            if node_id in self.tokens:
                token = self.tokens[node_id]
                if delta_expectation is not None:
                    token.expectation_value += delta_expectation
                if delta_pressure is not None:
                    token.pressure_value += delta_pressure
    
    def remove_attention(self, node_id: str):
        """移除注意令牌"""
        with self._lock:
            if node_id in self.tokens:
                del self.tokens[node_id]
    
    def decay_all(self, dt: float):
        """所有注意令牌衰减"""
        with self._lock:
            # 收集需要删除的键
            to_remove = []
            for token in self.tokens.values():
                token.decay(dt)
                # 记录需要移除的过小值
                if token.absolute_value < 0.1:
                    to_remove.append(token.node_id)
            
            # 遍历完成后删除
            for node_id in to_remove:
                del self.tokens[node_id]
    
    def get_strongest_attentions(self, n: int = 10) -> List[AttentionToken]:
        """获取最强的n个注意"""
        with self._lock:
            sorted_tokens = sorted(
                self.tokens.values(),
                key=lambda x: x.absolute_value,
                reverse=True
            )
            return sorted_tokens[:n]
    
    def _evict_weakest(self):
        """淘汰最弱的注意"""
        if self.tokens:
            weakest_id = min(
                self.tokens.items(), 
                key=lambda x: x[1].absolute_value
            )[0]
            del self.tokens[weakest_id]
    
    def size(self) -> int:
        """获取池大小"""
        with self._lock:
            return len(self.tokens)


class ActionActivationPool:
    """行动激活池"""
    
    def __init__(self):
        self.max_size = Config.MAX_ACTION_POOL_SIZE
        self.tokens: Dict[str, ActionToken] = {}
        self._lock = threading.Lock()
    
    def add_action(self, node_id: str, weight: float, action_type: str):
        """添加行动令牌"""
        with self._lock:
            if len(self.tokens) >= self.max_size:
                self._evict_weakest()
            
            token = ActionToken(
                node_id=node_id,
                weight=weight,
                action_type=action_type
            )
            self.tokens[node_id] = token
    
    def get_action(self, node_id: str) -> Optional[ActionToken]:
        """获取行动令牌"""
        with self._lock:
            return self.tokens.get(node_id)
    
    def update_weight(self, node_id: str, delta: float):
        """更新行动权重"""
        with self._lock:
            if node_id in self.tokens:
                self.tokens[node_id].update_weight(delta)
    
    def remove_action(self, node_id: str):
        """移除行动令牌"""
        with self._lock:
            if node_id in self.tokens:
                del self.tokens[node_id]
    
    def get_ready_actions(self, emotion_arousal: float = 0.0) -> List[ActionToken]:
        """获取准备执行的行动"""
        with self._lock:
            # 动态阈值 = 基础阈值 - λ * A
            threshold = Config.ACTION_BASE_THRESHOLD - Config.ACTION_AROUSAL_FACTOR * emotion_arousal
            
            ready_actions = [
                token for token in self.tokens.values()
                if token.weight >= threshold
            ]
            
            # 按权重排序
            return sorted(ready_actions, key=lambda x: x.weight, reverse=True)
    
    def _evict_weakest(self):
        """淘汰最弱的行动"""
        if self.tokens:
            weakest_id = min(self.tokens.items(), key=lambda x: x[1].weight)[0]
            del self.tokens[weakest_id]
    
    def size(self) -> int:
        """获取池大小"""
        with self._lock:
            return len(self.tokens)


class PoolManager:
    """激活池管理器"""
    
    def __init__(self):
        self.explicit_pool = ExplicitActivationPool()
        self.implicit_pool = ImplicitActivationPool()
        self.attention_pool = AttentionActivationPool()
        self.action_pool = ActionActivationPool()
        self.last_decay_time = time.time()
    
    def add_to_explicit(self, token: ActivationToken):
        """添加到显性池"""
        self.explicit_pool.add_token(token)
    
    def add_to_implicit(self, token: ActivationToken):
        """添加到隐性池"""
        self.implicit_pool.add_token(token)
    
    def add_attention(self, node_id: str, expectation: float = None, 
                     pressure: float = None, precursors: List[str] = None):
        """添加注意"""
        self.attention_pool.add_attention(node_id, expectation, pressure, precursors)
    
    def add_action(self, node_id: str, weight: float, action_type: str):
        """添加行动"""
        self.action_pool.add_action(node_id, weight, action_type)
    
    def decay_all(self):
        """所有池衰减"""
        current_time = time.time()
        dt = current_time - self.last_decay_time
        self.last_decay_time = current_time
        
        self.explicit_pool.decay_all(dt)
        self.implicit_pool.decay_all(dt)
        self.attention_pool.decay_all(dt)
    
    def get_pool_summary(self) -> Dict[str, Dict]:
        """获取所有池的摘要"""
        return {
            "explicit": {
                "size": self.explicit_pool.size(),
                "top_tokens": [(t.content, t.weight) for t in self.explicit_pool.get_top_tokens(5)]
            },
            "implicit": {
                "size": self.implicit_pool.size(),
                "top_tokens": [(t.content, t.weight) for t in self.implicit_pool.get_top_tokens(5)]
            },
            "attention": {
                "size": self.attention_pool.size(),
                "strongest": [(t.node_id, t.absolute_value) for t in self.attention_pool.get_strongest_attentions(5)]
            },
            "action": {
                "size": self.action_pool.size(),
                "actions": [(t.node_id, t.weight) for t in self.action_pool.tokens.values()]
            }
        }
    
    def clear_all(self):
        """清空所有池"""
        self.explicit_pool.clear()
        self.implicit_pool.clear()
        self.attention_pool.tokens.clear()
        self.action_pool.tokens.clear()