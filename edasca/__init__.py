"""
EDASCA - 情绪驱动激活扩散认知架构
Emotion-Driven Activation Spreading Cognitive Architecture

基于文档实现的Python版本
"""

from .core.edasca import EDASCA
from .core.config import Config

__version__ = "1.0.0"
__author__ = "EDASCA Implementation"

__all__ = ["EDASCA", "Config"]