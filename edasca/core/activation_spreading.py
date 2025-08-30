"""
激活扩散算法
"""

import numpy as np
from typing import List, Dict, Set, Optional, Tuple
from datetime import datetime
import math

from ..core.data_structures import ActivationToken, Origin, ConceptNode
from ..pools.pool_manager import PoolManager
from ..database.graph_db import GraphDatabase
from ..core.config import Config


class ActivationSpreading:
    """激活扩散引擎"""
    
    def __init__(self, pool_manager: PoolManager, graph_db: GraphDatabase):
        self.pool_manager = pool_manager
        self.graph_db = graph_db
        self.visited_nodes: Set[str] = set()  # 防止循环激活
    
    def spread_activation(self, source_node_id: str, current_emotion: np.ndarray, 
                         depth: int = 0, max_depth: int = None) -> None:
        """执行激活扩散
        
        Args:
            source_node_id: 源节点ID
            current_emotion: 当前情绪状态(PAD)
            depth: 当前扩散深度
            max_depth: 最大扩散深度
        """
        if max_depth is None:
            max_depth = Config.SPREAD_DEPTH_LIMIT
        
        # 防止过度扩散
        if depth >= max_depth:
            return
        
        # 防止循环
        if source_node_id in self.visited_nodes:
            return
        self.visited_nodes.add(source_node_id)
        
        # 获取源节点
        source_node = self.graph_db.get_node(source_node_id)
        if not source_node:
            return
        
        # 计算基础刺激强度
        S_base = 1.0 + current_emotion[2]  # 1 + D值
        
        # 获取所有出边
        out_edges = self.graph_db.get_out_edges(source_node_id)
        
        # 计算每个下游节点的激活增量
        activations = []
        for edge in out_edges:
            # 计算链接强度
            link_strength = self._calculate_link_strength(edge, current_emotion)
            
            # 计算激活增量
            delta_activation = S_base * link_strength
            
            activations.append((edge.target_id, delta_activation))
        
        # 应用激活
        for target_id, delta in activations:
            # 更新隐性激活池
            self._apply_activation(target_id, delta, source_node.origin, current_emotion)
            
            # 检查是否需要触发注意机制
            edge = self.graph_db.get_edge(source_node_id, target_id)
            if edge and abs(edge.recent_emotion_delta_ema[0]) > Config.ATTENTION_THRESHOLD:
                self._trigger_attention(target_id, edge.recent_emotion_delta_ema, [source_node_id])
            
            # 递归扩散
            self.spread_activation(target_id, current_emotion, depth + 1, max_depth)
        
        # 清除访问记录
        self.visited_nodes.discard(source_node_id)
    
    def _calculate_link_strength(self, edge, current_emotion: np.ndarray) -> float:
        """计算链接强度"""
        # 计算时间差
        current_time = datetime.now()
        time_delta = (current_time - edge.last_updated).total_seconds()
        
        # 基础链接强度公式
        # L(t) = (freq / (n + freq)) * (1 + k * ema_ΔP) * 
        #         (1 - avg_time_delta / (m + avg_time_delta)) * 
        #         (p + q * Δt') / (q * Δt')
        
        freq_factor = edge.frequency / (Config.FREQ_NORM_COEFF + edge.frequency)
        
        # 情绪影响因子
        emotion_factor = 1 + Config.EMOTION_IMPACT_COEFF * edge.recent_emotion_delta_ema[0]
        
        # 时间衰减因子
        time_decay_factor = 1 - edge.avg_time_delta / (Config.TIME_DECAY_COEFF + edge.avg_time_delta)
        
        # 新近度因子
        if time_delta > 0:
            recency_factor = (Config.RECENCY_P + Config.RECENCY_Q * time_delta) / (Config.RECENCY_Q * time_delta)
        else:
            recency_factor = 1.0
        
        link_strength = freq_factor * emotion_factor * time_decay_factor * recency_factor
        
        return max(0.0, link_strength)
    
    def _apply_activation(self, node_id: str, delta: float, origin: Origin, 
                         emotion_snapshot: np.ndarray) -> None:
        """应用激活到节点"""
        # 获取或创建激活令牌
        existing_token = self.pool_manager.implicit_pool.get_token(node_id)
        
        if existing_token:
            # 更新现有令牌
            existing_token.weight += delta
            existing_token.emotion_snapshot = emotion_snapshot.copy()
        else:
            # 获取节点信息
            node = self.graph_db.get_node(node_id)
            if node:
                # 创建新令牌
                token = ActivationToken(
                    node_id=node_id,
                    content=node.content,
                    weight=max(0.0, node.base_weight + delta),
                    origin=origin,
                    emotion_snapshot=emotion_snapshot.copy()
                )
                self.pool_manager.add_to_implicit(token)
    
    def _trigger_attention(self, node_id: str, emotion_delta: np.ndarray, 
                          precursors: List[str]) -> None:
        """触发注意机制"""
        # 根据情绪变化决定是期待还是压力
        if emotion_delta[0] > 0:  # 正向变化
            expectation = min(1.0, abs(emotion_delta[0]))
            pressure = 0.0
        else:  # 负向变化
            expectation = 0.0
            pressure = max(-1.0, -abs(emotion_delta[0]))
        
        # 添加到注意池
        self.pool_manager.add_attention(
            node_id=node_id,
            expectation=expectation,
            pressure=pressure,
            precursors=precursors
        )
        
        # 注意节点在隐性池中的衰减变慢
        token = self.pool_manager.implicit_pool.get_token(node_id)
        if token:
            token.decay_rate /= (1 + abs(expectation or pressure))
    
    def batch_spread(self, node_ids: List[str], current_emotion: np.ndarray) -> None:
        """批量激活扩散"""
        self.visited_nodes.clear()
        
        for node_id in node_ids:
            if node_id not in self.visited_nodes:
                self.spread_activation(node_id, current_emotion)
    
    def calculate_activation_flow(self, source_id: str, target_id: str, 
                                 current_emotion: np.ndarray) -> float:
        """计算从源节点到目标节点的激活流量"""
        # 使用广度优先搜索计算路径上的激活流量
        from collections import deque
        
        queue = deque([(source_id, 1.0)])  # (node_id, accumulated_flow)
        visited = set([source_id])
        total_flow = 0.0
        
        while queue:
            current_id, flow = queue.popleft()
            
            if current_id == target_id:
                total_flow += flow
                continue
            
            # 获取所有出边
            out_edges = self.graph_db.get_out_edges(current_id)
            
            for edge in out_edges:
                if edge.target_id not in visited:
                    visited.add(edge.target_id)
                    
                    # 计算这条边的流量
                    edge_strength = self._calculate_link_strength(edge, current_emotion)
                    new_flow = flow * edge_strength
                    
                    if new_flow > 0.01:  # 只考虑有意义的流量
                        queue.append((edge.target_id, new_flow))
        
        return total_flow