"""
图数据库管理器
"""

import os
import json
import pickle
from typing import Dict, List, Optional, Tuple, Set
from datetime import datetime
import numpy as np
from collections import defaultdict
import threading

from ..core.data_structures import (
    ConceptNode, RelationEdge, SensoryMemoryNode,
    Origin, NodeType
)
from ..core.config import Config


class GraphDatabase:
    """图数据库管理器"""
    
    def __init__(self, db_path: str = None):
        self.db_path = db_path or Config.DB_PATH
        self.nodes: Dict[str, ConceptNode] = {}
        self.edges: Dict[str, RelationEdge] = {}
        self.sensory_memories: Dict[str, SensoryMemoryNode] = {}
        self.content_index: Dict[str, Set[str]] = defaultdict(set)  # 内容到节点ID的映射
        self._lock = threading.RLock()
        
        # 确保数据库目录存在
        os.makedirs(self.db_path, exist_ok=True)
        
        # 尝试加载已有数据
        self._load_data()
    
    def _get_edge_key(self, source_id: str, target_id: str) -> str:
        """生成边的键"""
        return f"{source_id}->{target_id}"
    
    def add_node(self, node: ConceptNode) -> str:
        """添加概念节点"""
        with self._lock:
            # 检查是否已存在相同内容的节点
            existing_ids = self.content_index.get(node.content.lower(), set())
            for existing_id in existing_ids:
                existing_node = self.nodes.get(existing_id)
                if existing_node and existing_node.origin == node.origin:
                    # 更新现有节点
                    existing_node.update_base_weight()
                    return existing_id
            
            # 添加新节点
            self.nodes[node.id] = node
            self.content_index[node.content.lower()].add(node.id)
            self._save_data()
            return node.id
    
    def get_node(self, node_id: str) -> Optional[ConceptNode]:
        """获取节点"""
        with self._lock:
            return self.nodes.get(node_id)
    
    def get_node_by_content(self, content: str, origin: Origin = None) -> Optional[ConceptNode]:
        """根据内容获取节点"""
        with self._lock:
            node_ids = self.content_index.get(content.lower(), set())
            for node_id in node_ids:
                node = self.nodes.get(node_id)
                if node and (origin is None or node.origin == origin):
                    return node
            return None
    
    def add_edge(self, source_id: str, target_id: str, 
                 time_delta: float = 0.0, emotion_delta: np.ndarray = None) -> str:
        """添加关系边"""
        with self._lock:
            edge_key = self._get_edge_key(source_id, target_id)
            
            if edge_key in self.edges:
                # 更新现有边
                edge = self.edges[edge_key]
                edge.update_frequency()
                # 更新平均时间间隔
                edge.avg_time_delta = (edge.avg_time_delta * (edge.frequency - 1) + time_delta) / edge.frequency
                # 更新情绪变化
                if emotion_delta is not None:
                    edge.update_emotion_ema(emotion_delta)
            else:
                # 创建新边
                edge = RelationEdge(
                    source_id=source_id,
                    target_id=target_id,
                    avg_time_delta=time_delta
                )
                if emotion_delta is not None:
                    edge.update_emotion_ema(emotion_delta)
                self.edges[edge_key] = edge
            
            self._save_data()
            return edge_key
    
    def get_edge(self, source_id: str, target_id: str) -> Optional[RelationEdge]:
        """获取边"""
        with self._lock:
            edge_key = self._get_edge_key(source_id, target_id)
            return self.edges.get(edge_key)
    
    def get_out_edges(self, node_id: str) -> List[RelationEdge]:
        """获取节点的出边"""
        with self._lock:
            return [edge for edge in self.edges.values() if edge.source_id == node_id]
    
    def get_in_edges(self, node_id: str) -> List[RelationEdge]:
        """获取节点的入边"""
        with self._lock:
            return [edge for edge in self.edges.values() if edge.target_id == node_id]
    
    def add_sensory_memory(self, memory: SensoryMemoryNode) -> str:
        """添加感受记忆"""
        with self._lock:
            self.sensory_memories[memory.id] = memory
            
            # 检查是否需要执行遗忘
            if len(self.sensory_memories) > Config.MAX_MEMORIES:
                self._forget_memories()
            
            self._save_data()
            return memory.id
    
    def get_sensory_memory(self, memory_id: str) -> Optional[SensoryMemoryNode]:
        """获取感受记忆"""
        with self._lock:
            return self.sensory_memories.get(memory_id)
    
    def find_sensory_memories_by_time(self, center_time: datetime, 
                                     time_window: float) -> List[SensoryMemoryNode]:
        """根据时间查找感受记忆"""
        with self._lock:
            memories = []
            for memory in self.sensory_memories.values():
                time_diff = abs((memory.timestamp - center_time).total_seconds())
                if time_diff <= time_window:
                    memories.append(memory)
            
            # 按时间差排序
            memories.sort(key=lambda m: abs((m.timestamp - center_time).total_seconds()))
            return memories
    
    def find_sensory_memories_by_emotion(self, emotion: np.ndarray, 
                                        similarity_threshold: float = 0.5) -> List[SensoryMemoryNode]:
        """根据情绪相似度查找感受记忆"""
        with self._lock:
            memories = []
            for memory in self.sensory_memories.values():
                # 计算余弦相似度
                similarity = np.dot(emotion, memory.emotion_at_encoding) / (
                    np.linalg.norm(emotion) * np.linalg.norm(memory.emotion_at_encoding)
                )
                if similarity >= similarity_threshold:
                    memories.append((memory, similarity))
            
            # 按相似度排序
            memories.sort(key=lambda x: x[1], reverse=True)
            return [m[0] for m in memories]
    
    def _forget_memories(self):
        """执行遗忘机制"""
        memories_to_forget = []
        
        # 获取当前记忆数量，避免在遍历过程中计算
        current_memory_count = len(self.sensory_memories)
        
        for memory in self.sensory_memories.values():
            # 计算遗忘概率
            age_in_days = (datetime.now() - memory.timestamp).days / 365.25
            age_factor = np.log(1 + age_in_days) / 10
            
            P_forget = (Config.BASE_FORGET_PROB * 
                       (1 - memory.importance) * 
                       (1 + (current_memory_count / Config.MAX_MEMORIES) ** Config.FORGET_PRESSURE_K) * 
                       age_factor)
            
            if np.random.random() < P_forget:
                memories_to_forget.append(memory.id)
        
        # 执行遗忘 - 确保不会删除太多
        max_to_forget = max(0, current_memory_count - Config.MAX_MEMORIES)
        for memory_id in memories_to_forget[:max_to_forget]:
            if memory_id in self.sensory_memories:  # 双重检查
                del self.sensory_memories[memory_id]
    
    def _save_data(self):
        """保存数据到磁盘"""
        try:
            data = {
                "nodes": {k: self._node_to_dict(v) for k, v in self.nodes.items()},
                "edges": {k: self._edge_to_dict(v) for k, v in self.edges.items()},
                "sensory_memories": {k: self._sensory_to_dict(v) for k, v in self.sensory_memories.items()},
                "content_index": {k: list(v) for k, v in self.content_index.items()}
            }
            
            with open(os.path.join(self.db_path, "edasca_data.pkl"), "wb") as f:
                pickle.dump(data, f)
        except Exception as e:
            print(f"保存数据失败: {e}")
    
    def _load_data(self):
        """从磁盘加载数据"""
        try:
            file_path = os.path.join(self.db_path, "edasca_data.pkl")
            if os.path.exists(file_path):
                with open(file_path, "rb") as f:
                    data = pickle.load(f)
                
                self.nodes = {k: self._dict_to_node(v) for k, v in data.get("nodes", {}).items()}
                self.edges = {k: self._dict_to_edge(v) for k, v in data.get("edges", {}).items()}
                self.sensory_memories = {k: self._dict_to_sensory(v) for k, v in data.get("sensory_memories", {}).items()}
                self.content_index = defaultdict(set, {k: set(v) for k, v in data.get("content_index", {}).items()})
        except Exception as e:
            print(f"加载数据失败: {e}")
    
    def _node_to_dict(self, node: ConceptNode) -> Dict:
        """节点转字典"""
        return {
            "id": node.id,
            "content": node.content,
            "type": node.type.value,
            "origin": node.origin.value,
            "base_weight": node.base_weight,
            "last_activated": node.last_activated.isoformat(),
            "emotion_ema": node.emotion_ema.tolist()
        }
    
    def _dict_to_node(self, data: Dict) -> ConceptNode:
        """字典转节点"""
        node = ConceptNode(
            id=data["id"],
            content=data["content"],
            type=NodeType(data["type"]),
            origin=Origin(data["origin"]),
            base_weight=data["base_weight"],
            last_activated=datetime.fromisoformat(data["last_activated"]),
            emotion_ema=np.array(data["emotion_ema"])
        )
        return node
    
    def _edge_to_dict(self, edge: RelationEdge) -> Dict:
        """边转字典"""
        return {
            "source_id": edge.source_id,
            "target_id": edge.target_id,
            "frequency": edge.frequency,
            "avg_time_delta": edge.avg_time_delta,
            "recent_emotion_delta_ema": edge.recent_emotion_delta_ema.tolist(),
            "ema_alpha": edge.ema_alpha,
            "last_updated": edge.last_updated.isoformat()
        }
    
    def _dict_to_edge(self, data: Dict) -> RelationEdge:
        """字典转边"""
        edge = RelationEdge(
            source_id=data["source_id"],
            target_id=data["target_id"],
            frequency=data["frequency"],
            avg_time_delta=data["avg_time_delta"],
            ema_alpha=data["ema_alpha"],
            last_updated=datetime.fromisoformat(data["last_updated"])
        )
        edge.recent_emotion_delta_ema = np.array(data["recent_emotion_delta_ema"])
        return edge
    
    def _sensory_to_dict(self, memory: SensoryMemoryNode) -> Dict:
        """感受记忆转字典"""
        return {
            "id": memory.id,
            "content": memory.content,
            "origin": memory.origin.value,
            "timestamp": memory.timestamp.isoformat(),
            "emotion_at_encoding": memory.emotion_at_encoding.tolist(),
            "importance": memory.importance,
            "links": memory.links
        }
    
    def _dict_to_sensory(self, data: Dict) -> SensoryMemoryNode:
        """字典转感受记忆"""
        memory = SensoryMemoryNode(
            id=data["id"],
            content=data["content"],
            origin=Origin(data["origin"]),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            emotion_at_encoding=np.array(data["emotion_at_encoding"]),
            importance=data["importance"],
            links=data["links"]
        )
        return memory
    
    def get_statistics(self) -> Dict:
        """获取数据库统计信息"""
        with self._lock:
            return {
                "node_count": len(self.nodes),
                "edge_count": len(self.edges),
                "sensory_memory_count": len(self.sensory_memories),
                "avg_node_weight": np.mean([n.base_weight for n in self.nodes.values()]) if self.nodes else 0,
                "avg_edge_frequency": np.mean([e.frequency for e in self.edges.values()]) if self.edges else 0
            }