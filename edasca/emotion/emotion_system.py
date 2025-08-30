"""
情绪系统（PAD模型）
"""

import numpy as np
import threading
import time
from typing import Dict, List, Tuple, Optional
from enum import Enum

from ..core.data_structures import EmotionPAD
from ..core.config import Config


class EmotionLabel(Enum):
    """情绪标签"""
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    FEARFUL = "fearful"
    DISGUSTED = "disgusted"
    SURPRISED = "surprised"
    CALM = "calm"
    EXCITED = "excited"
    BORED = "bored"
    RELAXED = "relaxed"
    NERVOUS = "nervous"


class EmotionClassifier:
    """PAD情绪分类器"""
    
    def __init__(self):
        # 定义基本情绪的PAD值（基于Mehrabian的研究）
        self.emotion_templates = {
            EmotionLabel.HAPPY: np.array([0.7, 0.5, 0.6]),
            EmotionLabel.SAD: np.array([-0.6, -0.3, -0.4]),
            EmotionLabel.ANGRY: np.array([-0.5, 0.7, 0.3]),
            EmotionLabel.FEARFUL: np.array([-0.4, 0.6, -0.5]),
            EmotionLabel.DISGUSTED: np.array([-0.6, 0.3, -0.2]),
            EmotionLabel.SURPRISED: np.array([0.2, 0.8, -0.1]),
            EmotionLabel.CALM: np.array([0.3, -0.5, 0.4]),
            EmotionLabel.EXCITED: np.array([0.5, 0.8, 0.2]),
            EmotionLabel.BORED: np.array([-0.3, -0.7, -0.2]),
            EmotionLabel.RELAXED: np.array([0.6, -0.4, 0.5]),
            EmotionLabel.NERVOUS: np.array([-0.2, 0.6, -0.3])
        }
    
    def classify(self, pad_values: np.ndarray) -> Tuple[EmotionLabel, float]:
        """将PAD值分类为情绪标签"""
        best_label = EmotionLabel.CALM
        best_similarity = -1.0
        
        for label, template in self.emotion_templates.items():
            # 计算余弦相似度
            template_norm = np.linalg.norm(template)
            pad_norm = np.linalg.norm(pad_values)
            
            # 避免除零错误
            if template_norm > 0 and pad_norm > 0:
                similarity = np.dot(pad_values, template) / (template_norm * pad_norm)
            else:
                similarity = 0
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_label = label
        
        return best_label, best_similarity


class EmotionSystem:
    """情绪系统管理器"""
    
    def __init__(self):
        self.current_emotion = EmotionPAD()
        self.baseline_emotion = Config.PAD_BASELINE
        self.classifier = EmotionClassifier()
        self.correctness = 0.0  # 正确感
        self.incongruity = 0.0  # 违和感
        self.predictability = 0.0  # 预测可预测性
        self._lock = threading.Lock()
        self.last_update_time = time.time()
    
    def update_emotion(self, dominance: float = None, 
                      correctness_delta: float = None,
                      surprise: float = None,
                      predictability_delta: float = None) -> None:
        """更新情绪状态"""
        with self._lock:
            # P值更新：P += α * D + β * 匹配权重 * 正确感 + γ * (期待值 - 压力值)
            if dominance is not None:
                delta_p = (Config.P_ALPHA * dominance + 
                          Config.P_BETA * correctness_delta * 0.1)
                self.current_emotion.p += delta_p
            
            # A值更新：A += δ * 预测意外性 + ε * (|期待值| + |压力值|)
            if surprise is not None:
                delta_a = (Config.A_DELTA * surprise + 
                          Config.A_EPSILON * abs(correctness_delta or 0) * 0.1)
                self.current_emotion.a += delta_a
            
            # D值更新：D += ζ * 预测成功率
            if predictability_delta is not None:
                delta_d = Config.D_ZETA * predictability_delta * 0.1
                self.current_emotion.d += delta_d
            
            # 应用回归函数
            self._apply_regression()
            
            # 限制范围
            self._clamp_values()
            
            self.last_update_time = time.time()
    
    def _apply_regression(self) -> None:
        """应用回归函数"""
        # 回归量 = (基准值 - 当前值) * μ * (1 + |当前值 - 基准值|^2)
        for i, (current, baseline) in enumerate(zip(
            [self.current_emotion.p, self.current_emotion.a, self.current_emotion.d],
            self.baseline_emotion
        )):
            regression = (baseline - current) * Config.REGRESSION_MU * (
                1 + abs(current - baseline) ** 2
            )
            
            if i == 0:
                self.current_emotion.p += regression
            elif i == 1:
                self.current_emotion.a += regression
            else:
                self.current_emotion.d += regression
    
    def _clamp_values(self) -> None:
        """限制情绪值在[-1,1]范围内"""
        self.current_emotion.p = max(-1.0, min(1.0, self.current_emotion.p))
        self.current_emotion.a = max(-1.0, min(1.0, self.current_emotion.a))
        self.current_emotion.d = max(-1.0, min(1.0, self.current_emotion.d))
    
    def update_correctness(self, matched: bool, strength: float = 0.1) -> None:
        """更新正确感/违和感"""
        with self._lock:
            if matched:
                self.correctness = min(1.0, self.correctness + strength)
                self.incongruity = max(-1.0, self.incongruity - strength)
            else:
                self.correctness = max(-1.0, self.correctness - strength)
                self.incongruity = min(1.0, self.incongruity + strength)
    
    def get_current_state(self) -> Dict:
        """获取当前情绪状态"""
        with self._lock:
            return {
                "pad": self.current_emotion.to_array().tolist(),
                "correctness": self.correctness,
                "incongruity": self.incongruity,
                "predictability": self.predictability,
                "emotion_label": self.classifier.classify(
                    self.current_emotion.to_array()
                )[0].value
            }
    
    def get_emotion_label(self) -> Tuple[EmotionLabel, float]:
        """获取当前情绪标签"""
        with self._lock:
            return self.classifier.classify(self.current_emotion.to_array())
    
    def reset_to_baseline(self) -> None:
        """重置到基线情绪"""
        with self._lock:
            self.current_emotion.p = self.baseline_emotion[0]
            self.current_emotion.a = self.baseline_emotion[1]
            self.current_emotion.d = self.baseline_emotion[2]
            self.correctness = 0.0
            self.incongruity = 0.0
            self.predictability = 0.0
    
    def apply_emotional_impact(self, delta_pad: np.ndarray) -> None:
        """应用情绪冲击"""
        with self._lock:
            self.current_emotion.update(
                delta_pad[0], delta_pad[1], delta_pad[2]
            )
            self._clamp_values()
    
    def calculate_emotional_distance(self, other_pad: np.ndarray) -> float:
        """计算与其他情绪状态的距离"""
        with self._lock:
            current = self.current_emotion.to_array()
            return np.linalg.norm(current - other_pad)
    
    def is_emotionally_intense(self, threshold: float = 0.5) -> bool:
        """判断情绪是否强烈"""
        with self._lock:
            pad = self.current_emotion.to_array()
            return any(abs(x) > threshold for x in pad)
    
    def get_emotion_vector(self) -> np.ndarray:
        """获取情绪向量"""
        with self._lock:
            return self.current_emotion.to_array().copy()