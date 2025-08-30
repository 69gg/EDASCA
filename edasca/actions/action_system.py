"""
行动系统
"""

import numpy as np
from typing import Dict, List, Optional, Callable, Any, Tuple
from datetime import datetime, timedelta
import threading
import time

from ..core.data_structures import ConceptNode, NodeType, Origin, ActivationToken
from ..pools.pool_manager import PoolManager
from ..database.graph_db import GraphDatabase
from ..emotion.emotion_system import EmotionSystem
from ..core.config import Config


class InternalAction:
    """内在行动基类"""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
    
    def execute(self, *args, **kwargs) -> Any:
        """执行行动"""
        raise NotImplementedError


class RecallAction(InternalAction):
    """主动回忆行动"""
    
    def __init__(self):
        super().__init__("recall", "主动回忆过去的经历")
    
    def execute(self, query_content: str, **kwargs) -> List[str]:
        """执行回忆"""
        # 获取参数
        graph_db = kwargs.get('graph_db')
        pool_manager = kwargs.get('pool_manager')
        
        # 解析时间线索
        from ..sensors.sensor_system import TimeSensor
        time_sensor = TimeSensor(graph_db, pool_manager)
        time_diff = time_sensor.parse_time_expression(query_content)
        
        if time_diff is not None:
            # 基于时间线索检索
            target_time = datetime.now() - timedelta(seconds=time_diff)
            memories = graph_db.find_sensory_memories_by_time(target_time, time_window)
        else:
            # 基于内容相关性检索
            memories = []
            for memory in graph_db.sensory_memories.values():
                if query_content in memory.content:
                    memories.append(memory)
        
        # 将回忆内容注入隐性池
        recalled_tokens = []
        for memory in memories[:5]:  # 限制回忆数量
            for node_id in memory.links:
                node = graph_db.get_node(node_id)
                if node:
                    token = ActivationToken(
                        node_id=node_id,
                        content=node.content,
                        weight=0.8,  # 回忆的权重较高
                        origin=Origin.INTERNAL,
                        emotion_snapshot=memory.emotion_at_encoding
                    )
                    pool_manager.add_to_implicit(token)
                    recalled_tokens.append(node.content)
        
        return recalled_tokens


class FocusOutwardAction(InternalAction):
    """主动注意外部行动"""
    
    def __init__(self):
        super().__init__("focus_outward", "将注意力转向外部")
    
    def execute(self, pool_manager: PoolManager, **kwargs) -> Dict[str, float]:
        """执行转向外部注意"""
        # 调整输入选择器的偏好
        # 这里通过增加外部输入的权重来实现
        adjustment = 0.5  # 外部输入增益
        
        # 获取当前显性池中的外部内容
        external_tokens = pool_manager.explicit_pool.get_tokens_by_origin(Origin.EXTERNAL)
        
        # 增加外部令牌的权重
        for token in external_tokens:
            pool_manager.explicit_pool.update_weight(token.node_id, adjustment)
        
        return {"external_gain": adjustment, "affected_tokens": len(external_tokens)}


class FocusInwardAction(InternalAction):
    """主动深思行动"""
    
    def __init__(self):
        super().__init__("focus_inward", "将注意力转向内部")
    
    def execute(self, pool_manager: PoolManager, **kwargs) -> Dict[str, float]:
        """执行转向内部注意"""
        # 调整输入选择器的偏好
        adjustment = 0.5  # 内部输入增益
        
        # 获取当前显性池中的内部内容
        internal_tokens = pool_manager.explicit_pool.get_tokens_by_origin(Origin.INTERNAL)
        
        # 增加内部令牌的权重
        for token in internal_tokens:
            pool_manager.explicit_pool.update_weight(token.node_id, adjustment)
        
        return {"internal_gain": adjustment, "affected_tokens": len(internal_tokens)}


class OrganizeThoughtsAction(InternalAction):
    """整理思绪行动"""
    
    def __init__(self):
        super().__init__("organize_thoughts", "整理当前思绪")
    
    def execute(self, pool_manager: PoolManager, **kwargs) -> Dict[str, Any]:
        """执行整理思绪"""
        # 获取注意池中的内容
        attentions = pool_manager.attention_pool.get_strongest_attentions()
        
        thoughts = []
        for attention in attentions:
            # 获取相关节点
            node = graph_db.get_node(attention.node_id)
            if node:
                thought = {
                    "content": node.content,
                    "expectation": attention.expectation_value,
                    "pressure": attention.pressure_value,
                    "precursors": attention.precursor_nodes
                }
                thoughts.append(thought)
        
        return {
            "thoughts": thoughts,
            "count": len(thoughts),
            "summary": f"当前有{len(thoughts)}个需要注意的思绪"
        }


class FeelStateAction(InternalAction):
    """感受状态行动"""
    
    def __init__(self):
        super().__init__("feel_state", "感受当前状态")
    
    def execute(self, emotion_system: EmotionSystem, **kwargs) -> Dict[str, Any]:
        """执行感受状态"""
        emotion_state = emotion_system.get_current_state()
        
        return {
            "emotion": emotion_state["emotion_label"],
            "pad_values": emotion_state["pad"],
            "correctness": emotion_state["correctness"],
            "incongruity": emotion_state["incongruity"]
        }


class ExternalAction:
    """外在行动基类"""
    
    def __init__(self, name: str, action_type: str, description: str):
        self.name = name
        self.action_type = action_type
        self.description = description
    
    def execute(self, *args, **kwargs) -> Any:
        """执行行动"""
        raise NotImplementedError


class CommunicateAction(ExternalAction):
    """沟通行动"""
    
    def __init__(self):
        super().__init__("communicate", "communication", "生成语言回复")
    
    def execute(self, content: str, pool_manager: PoolManager, graph_db: GraphDatabase, emotion_system: EmotionSystem) -> str:
        """执行沟通"""
        # 将回复内容添加到显性池
        from ..core.data_structures import ConceptNode, NodeType
        node = ConceptNode(
            content=content,
            type=NodeType.WORD,
            origin=Origin.INTERNAL
        )
        node_id = graph_db.add_node(node)
        
        token = ActivationToken(
            node_id=node_id,
            content=content,
            weight=0.9,
            origin=Origin.INTERNAL,
            emotion_snapshot=emotion_system.get_emotion_vector()
        )
        pool_manager.add_to_explicit(token)
        
        return content


class SearchAction(ExternalAction):
    """搜索行动"""
    
    def __init__(self):
        super().__init__("search", "information_gathering", "搜索信息")
    
    def execute(self, query: str) -> List[str]:
        """执行搜索"""
        # 这里可以集成实际的搜索功能
        # 目前返回模拟结果
        return [f"搜索结果: {query}"]


class ActionSystem:
    """行动系统管理器"""
    
    def __init__(self, pool_manager: PoolManager, graph_db: GraphDatabase, 
                 emotion_system: EmotionSystem):
        self.pool_manager = pool_manager
        self.graph_db = graph_db
        self.emotion_system = emotion_system
        
        # 注册内在行动
        self.internal_actions: Dict[str, InternalAction] = {
            "recall": RecallAction(),
            "focus_outward": FocusOutwardAction(),
            "focus_inward": FocusInwardAction(),
            "organize_thoughts": OrganizeThoughtsAction(),
            "feel_state": FeelStateAction()
        }
        
        # 注册外在行动
        self.external_actions: Dict[str, ExternalAction] = {
            "communicate": CommunicateAction(),
            "search": SearchAction()
        }
        
        self.action_history: List[Dict] = []
        self._lock = threading.Lock()
    
    def trigger_internal_action(self, action_name: str, *args, **kwargs) -> Any:
        """触发内在行动"""
        with self._lock:
            if action_name in self.internal_actions:
                action = self.internal_actions[action_name]
                result = action.execute(*args, **kwargs, 
                                       graph_db=self.graph_db,
                                       pool_manager=self.pool_manager,
                                       emotion_system=self.emotion_system)
                
                # 记录行动历史
                self._record_action(action_name, "internal", args, kwargs, result)
                
                # 行动执行后更新情绪
                self._update_emotion_after_action(action_name, result)
                
                return result
            else:
                raise ValueError(f"未知的内在行动: {action_name}")
    
    def trigger_external_action(self, action_name: str, *args, **kwargs) -> Any:
        """触发外在行动"""
        with self._lock:
            if action_name in self.external_actions:
                action = self.external_actions[action_name]
                result = action.execute(*args, **kwargs, 
                                       pool_manager=self.pool_manager,
                                       graph_db=self.graph_db,
                                       emotion_system=self.emotion_system)
                
                # 记录行动历史
                self._record_action(action_name, "external", args, kwargs, result)
                
                # 行动执行后更新情绪
                self._update_emotion_after_action(action_name, result)
                
                return result
            else:
                raise ValueError(f"未知的外在行动: {action_name}")
    
    def check_ready_actions(self) -> List[Tuple[str, float]]:
        """检查准备执行的行动"""
        ready_actions = []
        
        # 获取当前情绪状态
        emotion_state = self.emotion_system.get_current_state()
        arousal = emotion_state["pad"][1]  # A值
        
        # 检查行动池中的行动
        action_tokens = self.pool_manager.action_pool.get_ready_actions(arousal)
        
        for token in action_tokens:
            ready_actions.append((token.action_type, token.weight))
        
        return ready_actions
    
    def execute_ready_actions(self) -> List[Dict]:
        """执行所有准备好的行动"""
        results = []
        
        for action_name, weight in self.check_ready_actions():
            try:
                # 判断是内在还是外在行动
                if action_name in self.internal_actions:
                    result = self.trigger_internal_action(action_name)
                elif action_name in self.external_actions:
                    result = self.trigger_external_action(action_name)
                else:
                    continue
                
                results.append({
                    "action": action_name,
                    "weight": weight,
                    "result": result,
                    "success": True
                })
                
                # 从行动池中移除已执行的行动
                self.pool_manager.action_pool.remove_action(action_name)
                
            except Exception as e:
                results.append({
                    "action": action_name,
                    "weight": weight,
                    "result": str(e),
                    "success": False
                })
        
        return results
    
    def _record_action(self, action_name: str, action_type: str, 
                      args: tuple, kwargs: dict, result: Any) -> None:
        """记录行动历史"""
        record = {
            "timestamp": datetime.now(),
            "action": action_name,
            "type": action_type,
            "args": args,
            "kwargs": kwargs,
            "result": result,
            "emotion_state": self.emotion_system.get_current_state()
        }
        self.action_history.append(record)
        
        # 限制历史记录数量
        if len(self.action_history) > 1000:
            self.action_history = self.action_history[-1000:]
    
    def _update_emotion_after_action(self, action_name: str, result: Any) -> None:
        """根据行动结果更新情绪"""
        # 这里可以根据行动结果调整情绪
        # 成功的行动增加愉悦度和支配度
        if isinstance(result, dict) and result.get("success", True):
            self.emotion_system.update_emotion(
                dominance=0.1,  # 增加支配感
                correctness_delta=0.2  # 增加正确感
            )
        else:
            self.emotion_system.update_emotion(
                dominance=-0.1,  # 减少支配感
                correctness_delta=-0.2  # 减少正确感
            )
    
    def get_action_statistics(self) -> Dict[str, Any]:
        """获取行动统计信息"""
        stats = {
            "internal_actions": len(self.internal_actions),
            "external_actions": len(self.external_actions),
            "total_executed": len(self.action_history),
            "recent_actions": self.action_history[-10:] if self.action_history else []
        }
        
        # 统计各行动的执行次数
        action_counts = {}
        for record in self.action_history:
            action_name = record["action"]
            action_counts[action_name] = action_counts.get(action_name, 0) + 1
        
        stats["action_counts"] = action_counts
        
        return stats