"""
加密服务抽象基类

定义加密服务的统一接口，支持 DCMH 哈希码加密和 CIR CNN 特征加密。
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import numpy as np
import torch


class BaseEncryptionService(ABC):
    """
    加密服务抽象基类。

    定义统一的加密服务接口，支持：
    - 密钥生成和加载
    - 数据库加密
    - 查询加密
    - 密文相似度计算
    """

    @abstractmethod
    def generate_keys(self) -> None:
        """
        生成加密密钥。

        必须在首次使用前调用。
        """
        pass

    @abstractmethod
    def load_keys(self) -> bool:
        """
        从持久化存储加载密钥。

        返回：
            是否成功加载密钥
        """
        pass

    @abstractmethod
    def save_keys(self) -> bool:
        """
        保存密钥到持久化存储。

        返回：
            是否成功保存密钥
        """
        pass

    @abstractmethod
    def encrypt_database(self, data: np.ndarray) -> np.ndarray:
        """
        加密数据库向量。

        参数：
            data: 明文数据库向量 [N, d]

        返回：
            加密后的数据库向量 [N, 2d]
        """
        pass

    @abstractmethod
    def encrypt_query(self, query: np.ndarray) -> np.ndarray:
        """
        加密查询向量。

        参数：
            query: 明文查询向量 [d] 或 [M, d]

        返回：
            加密后的查询向量 [2d] 或 [M, 2d]
        """
        pass

    @abstractmethod
    def compute_similarity(self,
                          encrypted_query: np.ndarray,
                          encrypted_db: np.ndarray) -> np.ndarray:
        """
        计算密文相似度。

        参数：
            encrypted_query: 加密查询向量 [M, 2d]
            encrypted_db: 加密数据库向量 [N, 2d]

        返回：
            相似度矩阵 [M, N]
        """
        pass

    @abstractmethod
    def compute_distance(self,
                        encrypted_query: np.ndarray,
                        encrypted_db: np.ndarray) -> np.ndarray:
        """
        计算密文距离。

        参数：
            encrypted_query: 加密查询向量 [M, 2d]
            encrypted_db: 加密数据库向量 [N, 2d]

        返回：
            距离矩阵 [M, N]
        """
        pass

    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """
        获取加密服务状态。

        返回：
            包含服务状态信息的字典
        """
        pass

    @abstractmethod
    def get_dim(self) -> int:
        """
        获取向量维度。

        返回：
            明文向量维度
        """
        pass

    @abstractmethod
    def get_encrypted_dim(self) -> int:
        """
        获取加密后向量维度。

        返回：
            加密后向量维度（通常是明文维度的 2 倍）
        """
        pass

    def has_keys(self) -> bool:
        """
        检查是否已有密钥。

        返回：
            是否已有密钥
        """
        return False

    def verify_consistency(self,
                          plaintext_data: np.ndarray,
                          num_samples: int = 10) -> Dict[str, Any]:
        """
        验证加密前后相似度一致性。

        参数：
            plaintext_data: 明文数据 [N, d]
            num_samples: 采样数量

        返回：
            包含验证结果的字典
        """
        return {
            "verified": False,
            "message": "Not implemented in base class"
        }