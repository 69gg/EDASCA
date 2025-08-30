"""
动态加权分词算法
"""

import re
from typing import List, Tuple, Dict, Set
from collections import defaultdict
import numpy as np

from ..core.data_structures import ActivationToken, Origin
from ..pools.pool_manager import PoolManager
from ..database.graph_db import GraphDatabase


class DynamicTokenizer:
    """动态加权分词器"""
    
    def __init__(self, pool_manager: PoolManager, graph_db: GraphDatabase):
        self.pool_manager = pool_manager
        self.graph_db = graph_db
        self.max_length = 50  # 最大词元长度
        
        # 构建动态词典
        self.dynamic_dict: Dict[str, Tuple[str, float, Origin]] = {}  # 内容 -> (节点ID, 权重, 起源)
        self._update_dynamic_dict()
    
    def _update_dynamic_dict(self):
        """更新动态词典"""
        self.dynamic_dict.clear()
        
        # 从所有激活池收集词元
        pools = [
            self.pool_manager.explicit_pool,
            self.pool_manager.implicit_pool,
            self.pool_manager.action_pool
        ]
        
        for pool in pools:
            if hasattr(pool, 'tokens'):
                for token in pool.tokens.values():
                    self.dynamic_dict[token.content.lower()] = (token.node_id, token.weight, token.origin)
        
        # 从注意池收集
        for node_id, attention in self.pool_manager.attention_pool.tokens.items():
            node = self.graph_db.get_node(node_id)
            if node:
                self.dynamic_dict[node.content.lower()] = (node_id, attention.absolute_value, node.origin)
    
    def tokenize(self, text: str, origin: Origin) -> List[ActivationToken]:
        """动态加权分词"""
        # 更新动态词典
        self._update_dynamic_dict()
        
        # 生成所有可能的分词方案
        candidates = self._generate_candidates(text, origin)
        
        if not candidates:
            # 如果没有候选方案，使用简单分词
            return self._simple_tokenize(text, origin)
        
        # 计算每个方案的权重
        scored_candidates = []
        for candidate in candidates:
            score = self._calculate_candidate_score(candidate)
            scored_candidates.append((candidate, score))
        
        # 选择最优方案
        best_candidate = max(scored_candidates, key=lambda x: x[1])[0]
        
        # 创建激活令牌
        tokens = []
        for word in best_candidate:
            node_id, weight, token_origin = self.dynamic_dict.get(word.lower(), (None, 0.1, origin))
            
            # 如果在词典中找不到，创建新节点
            if node_id is None:
                from ..core.data_structures import ConceptNode, NodeType
                node = ConceptNode(content=word, origin=origin)
                node_id = self.graph_db.add_node(node)
                weight = 0.1
            
            token = ActivationToken(
                node_id=node_id,
                content=word,
                weight=weight,
                origin=token_origin,
                emotion_snapshot=np.array([0.0, 0.0, 0.0])
            )
            tokens.append(token)
        
        return tokens
    
    def _generate_candidates(self, text: str, origin: Origin) -> List[List[str]]:
        """生成候选分词方案"""
        text = text.strip()
        if not text:
            return []
        
        # 简单的候选生成策略
        candidates = []
        
        # 1. 按最长匹配优先
        words = []
        i = 0
        while i < len(text):
            matched = False
            # 从最长到最短尝试匹配
            for length in range(min(self.max_length, len(text) - i), 0, -1):
                word = text[i:i+length]
                if word.lower() in self.dynamic_dict:
                    words.append(word)
                    i += length
                    matched = True
                    break
            
            if not matched:
                # 单字分词
                words.append(text[i])
                i += 1
        
        if words:
            candidates.append(words)
        
        # 2. 生成其他可能的分词方案
        # 这里可以添加更复杂的候选生成逻辑
        
        return candidates
    
    def _calculate_candidate_score(self, candidate: List[str]) -> float:
        """计算候选方案得分"""
        score = 0.0
        
        # 基础得分：匹配词的权重之和
        for word in candidate:
            if word.lower() in self.dynamic_dict:
                _, weight, _ = self.dynamic_dict[word.lower()]
                score += weight
            else:
                # 未匹配词的惩罚
                score -= 0.1
        
        # 长度惩罚：过长的分词方案得分较低
        length_penalty = len(candidate) * 0.05
        score -= length_penalty
        
        # 连贯性加分：相邻词在知识库中有连接的加分
        for i in range(len(candidate) - 1):
            word1, word2 = candidate[i], candidate[i+1]
            if word1.lower() in self.dynamic_dict and word2.lower() in self.dynamic_dict:
                node_id1, _, _ = self.dynamic_dict[word1.lower()]
                node_id2, _, _ = self.dynamic_dict[word2.lower()]
                
                edge = self.graph_db.get_edge(node_id1, node_id2)
                if edge:
                    score += edge.frequency * 0.01
        
        return score
    
    def _simple_tokenize(self, text: str, origin: Origin) -> List[ActivationToken]:
        """简单分词（回退方案）"""
        # 使用正则表达式进行简单分词
        words = re.findall(r'[\w]+|[^\w\s]', text)
        
        tokens = []
        for word in words:
            # 创建新节点
            from ..core.data_structures import ConceptNode, NodeType
            node = ConceptNode(content=word, origin=origin)
            node_id = self.graph_db.add_node(node)
            
            token = ActivationToken(
                node_id=node_id,
                content=word,
                weight=0.1,
                origin=origin,
                emotion_snapshot=np.array([0.0, 0.0, 0.0])
            )
            tokens.append(token)
        
        return tokens
    
    def get_word_importance(self, word: str) -> float:
        """获取词的重要性"""
        return self.dynamic_dict.get(word.lower(), (None, 0.1, None))[1]