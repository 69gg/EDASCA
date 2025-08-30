"""
环境配置管理
"""

import os
from dotenv import load_dotenv
from typing import Optional

load_dotenv()  # 加载.env文件


class Config:
    """配置类"""
    
    # OpenAI配置
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_BASE_URL: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
    
    # 情感分析提示
    EMOTION_ANALYSIS_PROMPT: str = os.getenv(
        "EMOTION_ANALYSIS_PROMPT",
        "请分析以下对话的情感色彩，返回PAD值（愉悦度-1到1，唤醒度-1到1，支配度-1到1）和情感标签："
    )
    
    # 模型保存路径
    MODEL_SAVE_PATH: str = os.getenv("MODEL_SAVE_PATH", "model/")
    
    @classmethod
    def validate(cls) -> bool:
        """验证配置"""
        if not cls.OPENAI_API_KEY:
            print("警告：未设置OPENAI_API_KEY")
            return False
        return True