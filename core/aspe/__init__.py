"""
ASPE (Asymmetric Scalar Product-Preserving Encryption) 加密方案

支持：
- ASPEForDCMH: DCMH 哈希码加密 (SkNN 风格)
- ASPEForCNN: CNN 特征向量加密 (SkNN 方案)
"""

from .dcmh_wrapper import ASPEForDCMH
from .cnn_wrapper import ASPEForCNN, create_cnn_aspe

__all__ = [
    'ASPEForDCMH',
    'ASPEForCNN',
    'create_cnn_aspe'
]