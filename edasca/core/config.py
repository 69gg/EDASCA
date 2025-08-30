"""
EDASCA配置参数
"""

from dataclasses import dataclass
from typing import Dict, Any
import uuid


@dataclass
class Config:
    """EDASCA系统配置"""
    
    # 激活池容量限制
    MAX_EXPLICIT_POOL_SIZE = 100
    MAX_IMPLICIT_POOL_SIZE = 1000
    MAX_ATTENTION_POOL_SIZE = 50
    MAX_ACTION_POOL_SIZE = 20
    
    # 情绪系统参数
    PAD_BASELINE = (0.0, 0.0, 0.0)  # P, A, D 基准值
    P_ALPHA = 0.1  # D对P的影响系数
    P_BETA = 0.3   # 正确感对P的影响系数
    P_GAMMA = 0.2  # 期待/压力对P的影响系数
    A_DELTA = 0.4  # 意外性对A的影响系数
    A_EPSILON = 0.1  # 注意张力对A的影响系数
    D_ZETA = 0.3   # 成功率对D的影响系数
    REGRESSION_MU = 0.05  # 回归速率系数
    
    # 激活扩散参数
    SPREAD_DEPTH_LIMIT = 3  # 最大扩散深度
    FREQ_NORM_COEFF = 4    # 频次标准化系数
    EMOTION_IMPACT_COEFF = 5  # 情绪影响系数
    TIME_DECAY_COEFF = 100  # 时间衰减系数
    RECENCY_P = 3          # 新近度参数p
    RECENCY_Q = 0.0001     # 新近度参数q
    
    # 基础权重更新参数
    WEIGHT_DECAY_GAMMA = 0.995  # 衰减因子
    WEIGHT_REWARD_ETA = 0.01    # 激活奖励值
    EMOTION_EMA_ALPHA = 0.3     # 情绪EMA平滑因子
    
    # 注意机制参数
    ATTENTION_THRESHOLD = 0.3    # 注意触发阈值
    ATTENTION_DECAY_RATE = 0.01 # 注意衰减速率
    
    # 行动系统参数
    ACTION_BASE_THRESHOLD = 0.7   # 行动基础阈值
    ACTION_AROUSAL_FACTOR = 0.2  # 唤醒度对行动阈值的影响系数
    
    # 感受器参数
    TIME_FUZZINESS_FACTOR = 0.5  # 时间模糊度系数
    EMOTION_INTENSITY_THRESHOLD = 0.5  # 情绪强度阈值
    EMOTION_CHANGE_THRESHOLD = 0.3    # 情绪变化阈值
    
    # 遗忘机制参数
    BASE_FORGET_PROB = 0.001    # 基础遗忘概率
    FORGET_PRESSURE_K = 2       # 容量压力放大系数
    MAX_MEMORIES = 50000        # 最大记忆容量
    
    # 线程参数
    EMOTION_UPDATE_INTERVAL = 0.1  # 情绪更新间隔(秒)
    
    # LLM参数
    LLM_MODEL_NAME = "gpt-3.5-turbo"  # 默认LLM模型
    LLM_TEMPERATURE = 0.7             # LLM温度参数
    
    # 数据库配置
    DB_PATH = "edasca_db"             # 数据库路径
    
    @classmethod
    def get_config_dict(cls) -> Dict[str, Any]:
        """获取配置字典"""
        config_dict = {}
        for key, value in cls.__dict__.items():
            if not key.startswith('_') and not callable(value):
                config_dict[key] = value
        return config_dict