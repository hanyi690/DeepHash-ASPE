"""
ASPE (Asymmetric Scalar Product-Preserving Encryption) 加密方案

支持：
- Scheme 1: 基础 2 级安全
- Scheme 2: 增强 3 级安全
- ASPEForDCMH: DCMH 哈希码加密
- ASPEForCNN: CNN 特征向量加密 (SkNN 方案)
"""

from .scheme1 import ASPEScheme1
from .scheme2 import ASPEScheme2
from .dcmh_wrapper import ASPEForDCMH
from .cnn_wrapper import ASPEForCNN, create_cnn_aspe

__all__ = [
    'ASPEScheme1',
    'ASPEScheme2',
    'ASPEForDCMH',
    'ASPEForCNN',
    'create_cnn_aspe'
]
