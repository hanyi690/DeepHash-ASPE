"""
DCMH 哈希码加密服务

封装 ASPEForDCMH，提供 DCMH 哈希码的加密功能。
支持密钥持久化、数据库加密、查询陷阱门生成。
"""

import numpy as np
import torch
from typing import Dict, Any, Optional
import logging

from app.services.base_encryption_service import BaseEncryptionService
from core.aspe.dcmh_wrapper import ASPEForDCMH
from app.services.key_manager import get_key_manager

logger = logging.getLogger(__name__)


class DCMHEncryptionService(BaseEncryptionService):
    """
    DCMH 哈希码加密服务。

    使用 ASPE (Asymmetric Scalar Product-Preserving Encryption) 方案
    加密 DCMH 生成的 {-1, +1} 哈希码。

    特性：
    - 密文内积 = 明文内积
    - 密文汉明距离 = 明文汉明距离
    - 支持密钥持久化
    """

    def __init__(self, bit_dim: int = 64, seed: int = 42):
        """
        初始化 DCMH 加密服务。

        参数：
            bit_dim: 哈希码位数（默认 64）
            seed: 随机种子（默认 42）
        """
        self.bit_dim = bit_dim
        self.seed = seed
        self.aspe: Optional[ASPEForDCMH] = None

        # 缓存的加密数据
        self._encrypted_database: Optional[np.ndarray] = None
        self._encrypted_text_database: Optional[np.ndarray] = None

        logger.info(f"[DCMHEncryptionService] 初始化：bit_dim={bit_dim}, seed={seed}")

    def generate_keys(self) -> None:
        """生成加密密钥。"""
        self.aspe = ASPEForDCMH(bit_dim=self.bit_dim, seed=self.seed)
        logger.info(f"[DCMHEncryptionService] 密钥已生成")

    def load_keys(self) -> bool:
        """
        从持久化存储加载密钥。

        返回：
            是否成功加载密钥
        """
        try:
            key_manager = get_key_manager()
            keys = key_manager.get_dcmh_keys(
                bit_dim=self.bit_dim,
                seed=self.seed
            )

            if keys:
                # 初始化 ASPE 实例
                self.aspe = ASPEForDCMH(bit_dim=self.bit_dim, seed=self.seed)

                # 应用密钥
                self.aspe.M1 = keys['M1']
                self.aspe.M2 = keys['M2']
                self.aspe.S = keys['S']
                if 'M1_inv' in keys and keys['M1_inv'] is not None:
                    self.aspe.M1_inv = keys['M1_inv']
                if 'M2_inv' in keys and keys['M2_inv'] is not None:
                    self.aspe.M2_inv = keys['M2_inv']

                logger.info("[DCMHEncryptionService] 已加载持久化密钥")
                return True
            else:
                # 生成新密钥并保存
                self.generate_keys()
                self.save_keys()
                return True

        except Exception as e:
            logger.warning(f"[DCMHEncryptionService] 加载密钥失败：{e}")
            self.generate_keys()
            return False

    def save_keys(self) -> bool:
        """
        保存密钥到持久化存储。

        返回：
            是否成功保存
        """
        try:
            if self.aspe is None:
                return False

            key_manager = get_key_manager()
            key_manager._dcmh_keys = {
                'M1': self.aspe.M1,
                'M2': self.aspe.M2,
                'S': self.aspe.S,
                'M1_inv': self.aspe.M1_inv,
                'M2_inv': self.aspe.M2_inv,
                'bit_dim': self.bit_dim,
                'seed': self.seed
            }
            key_manager._save_dcmh_keys()

            logger.info("[DCMHEncryptionService] 密钥已保存")
            return True

        except Exception as e:
            logger.error(f"[DCMHEncryptionService] 保存密钥失败：{e}")
            return False

    def _ensure_initialized(self) -> None:
        """确保服务已初始化。"""
        if self.aspe is None:
            self.load_keys()

    def encrypt_database(self, data: np.ndarray) -> np.ndarray:
        """
        加密数据库哈希码。

        参数：
            data: 明文哈希码 [N, bit_dim]

        返回：
            加密后的哈希码 [N, 2*bit_dim]
        """
        self._ensure_initialized()
        encrypted = self.aspe.GenEnc(data)
        self._encrypted_database = encrypted
        return encrypted

    def encrypt_text_database(self, text_codes: np.ndarray) -> np.ndarray:
        """
        加密文本哈希码数据库。

        参数：
            text_codes: 文本哈希码 [N, bit_dim]

        返回：
            加密后的哈希码 [N, 2*bit_dim]
        """
        self._ensure_initialized()
        encrypted = self.aspe.GenEnc(text_codes)
        self._encrypted_text_database = encrypted
        return encrypted

    def encrypt_query(self, query: np.ndarray) -> np.ndarray:
        """
        加密查询哈希码（生成陷阱门）。

        参数：
            query: 查询哈希码 [bit_dim] 或 [M, bit_dim]

        返回：
            加密后的查询 [2*bit_dim] 或 [M, 2*bit_dim]
        """
        self._ensure_initialized()

        if query.ndim == 1:
            query = query.reshape(1, -1)

        return self.aspe.GenTrap(query)

    def compute_similarity(self,
                          encrypted_query: np.ndarray,
                          encrypted_db: np.ndarray) -> np.ndarray:
        """
        计算密文相似度（内积）。

        参数：
            encrypted_query: 加密查询 [M, 2*bit_dim]
            encrypted_db: 加密数据库 [N, 2*bit_dim]

        返回：
            相似度矩阵 [M, N]（越高越相似）
        """
        self._ensure_initialized()
        d = self.bit_dim

        if encrypted_query.ndim == 1:
            encrypted_query = encrypted_query.reshape(1, -1)

        q1, q2 = encrypted_query[:, :d], encrypted_query[:, d:]
        r1, r2 = encrypted_db[:, :d], encrypted_db[:, d:]

        return q1 @ r1.T + q2 @ r2.T

    def compute_distance(self,
                        encrypted_query: np.ndarray,
                        encrypted_db: np.ndarray) -> np.ndarray:
        """
        计算密文汉明距离。

        参数：
            encrypted_query: 加密查询 [M, 2*bit_dim]
            encrypted_db: 加密数据库 [N, 2*bit_dim]

        返回：
            距离矩阵 [M, N]（越小越相似）
        """
        self._ensure_initialized()
        return self.aspe.ciphertext_hamming_distance(encrypted_query, encrypted_db)

    def get_status(self) -> Dict[str, Any]:
        """获取服务状态。"""
        return {
            "service_type": "dcmh_encryption",
            "bit_dim": self.bit_dim,
            "encrypted_dim": 2 * self.bit_dim,
            "seed": self.seed,
            "keys_loaded": self.aspe is not None,
            "scheme": "ASPE (SkNN 风格)",
            "database_encrypted": self._encrypted_database is not None,
            "database_size": self._encrypted_database.shape[0] if self._encrypted_database is not None else 0
        }

    def get_dim(self) -> int:
        """获取哈希码位数。"""
        return self.bit_dim

    def get_encrypted_dim(self) -> int:
        """获取加密后维度。"""
        return 2 * self.bit_dim

    def has_keys(self) -> bool:
        """检查是否已有密钥。"""
        return self.aspe is not None

    def verify_consistency(self,
                          plaintext_data: np.ndarray,
                          num_samples: int = 10) -> Dict[str, Any]:
        """
        验证加密前后汉明距离一致性。

        参数：
            plaintext_data: 明文哈希码 [N, bit_dim]
            num_samples: 采样数量

        返回：
            包含验证结果的字典
        """
        self._ensure_initialized()

        # 加密数据
        encrypted = self.encrypt_database(plaintext_data[:num_samples])

        # 计算明文汉明距离
        plain_inner = plaintext_data[:num_samples] @ plaintext_data[:num_samples].T
        plain_hamm = 0.5 * (self.bit_dim - plain_inner)

        # 计算密文汉明距离
        cipher_hamm = self.compute_distance(encrypted, encrypted)

        # 计算误差
        error = np.abs(plain_hamm - cipher_hamm).max()

        return {
            "verified": error < 1e-6,
            "max_error": float(error),
            "num_samples": num_samples
        }


# 全局服务实例
_dcmh_encryption_services: Dict[int, DCMHEncryptionService] = {}


def get_dcmh_encryption_service(bit_dim: int = 64, seed: int = 42) -> DCMHEncryptionService:
    """
    获取 DCMH 加密服务实例。

    参数：
        bit_dim: 哈希码位数
        seed: 随机种子

    返回：
        DCMHEncryptionService 实例
    """
    key = (bit_dim, seed)
    if key not in _dcmh_encryption_services:
        _dcmh_encryption_services[key] = DCMHEncryptionService(bit_dim=bit_dim, seed=seed)
    return _dcmh_encryption_services[key]