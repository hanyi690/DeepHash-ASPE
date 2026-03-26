"""
CIR CNN 特征加密服务

封装 ASPEForCNN，提供 CNN 特征的加密功能。
支持密钥持久化、数据库加密、查询陷阱门生成。
"""

import numpy as np
import torch
from typing import Dict, Any, Optional
import logging

from app.services.base_encryption_service import BaseEncryptionService
from core.aspe.cnn_wrapper import ASPEForCNN
from app.services.key_manager import get_key_manager

logger = logging.getLogger(__name__)


class CIREncryptionService(BaseEncryptionService):
    """
    CIR CNN 特征加密服务。

    使用 ASPE (Asymmetric Scalar Product-Preserving Encryption) 方案
    加密 CNN 提取的特征向量。

    特性：
    - 密文内积 = 明文内积
    - 支持 L2 归一化特征的余弦相似度计算
    - 支持密钥持久化
    """

    def __init__(self, feature_dim: int = 2048, device: Optional[str] = None):
        """
        初始化 CIR 加密服务。

        参数：
            feature_dim: 特征维度（默认 2048，对应 ResNet101-GeM）
            device: 计算设备
        """
        self.feature_dim = feature_dim
        self.device = torch.device(device) if device else torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.aspe: Optional[ASPEForCNN] = None

        # 缓存的加密数据
        self._encrypted_database: Optional[torch.Tensor] = None

        logger.info(f"[CIREncryptionService] 初始化：feature_dim={feature_dim}, device={self.device}")

    def generate_keys(self) -> None:
        """生成加密密钥。"""
        self.aspe = ASPEForCNN(feature_dim=self.feature_dim, device=str(self.device))
        self.aspe.generate_keys()
        logger.info(f"[CIREncryptionService] 密钥已生成")

    def load_keys(self) -> bool:
        """
        从持久化存储加载密钥。

        返回：
            是否成功加载密钥
        """
        try:
            key_manager = get_key_manager()
            keys = key_manager.get_cir_keys(feature_dim=self.feature_dim)

            if keys:
                # 初始化 ASPE 实例
                self.aspe = ASPEForCNN(feature_dim=self.feature_dim, device=str(self.device))

                # 应用密钥
                self.aspe.M1 = keys['M1'].to(self.device)
                self.aspe.M2 = keys['M2'].to(self.device)
                self.aspe.S = keys['S'].to(self.device)
                if 'M1_inv' in keys and keys['M1_inv'] is not None:
                    self.aspe.M1_inv = keys['M1_inv'].to(self.device)
                if 'M2_inv' in keys and keys['M2_inv'] is not None:
                    self.aspe.M2_inv = keys['M2_inv'].to(self.device)

                logger.info("[CIREncryptionService] 已加载持久化密钥")
                return True
            else:
                # 生成新密钥并保存
                self.generate_keys()
                self.save_keys()
                return True

        except Exception as e:
            logger.warning(f"[CIREncryptionService] 加载密钥失败：{e}")
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
            key_manager._cir_keys = {
                'M1': self.aspe.M1.cpu(),
                'M2': self.aspe.M2.cpu(),
                'S': self.aspe.S.cpu(),
                'M1_inv': self.aspe.M1_inv.cpu() if self.aspe.M1_inv is not None else None,
                'M2_inv': self.aspe.M2_inv.cpu() if self.aspe.M2_inv is not None else None,
                'feature_dim': self.feature_dim
            }
            key_manager._save_cir_keys()

            logger.info("[CIREncryptionService] 密钥已保存")
            return True

        except Exception as e:
            logger.error(f"[CIREncryptionService] 保存密钥失败：{e}")
            return False

    def _ensure_initialized(self) -> None:
        """确保服务已初始化。"""
        if self.aspe is None:
            self.load_keys()

    def encrypt_database(self, data: np.ndarray) -> np.ndarray:
        """
        加密数据库特征。

        参数：
            data: 明文特征 [N, feature_dim]

        返回：
            加密后的特征 [N, 2*feature_dim]
        """
        self._ensure_initialized()

        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)

        encrypted = self.aspe.encrypt_database(data)
        self._encrypted_database = encrypted

        # 返回 numpy 数组以保持接口一致性
        return encrypted.cpu().numpy()

    def encrypt_query(self, query: np.ndarray) -> np.ndarray:
        """
        加密查询特征（生成陷阱门）。

        参数：
            query: 查询特征 [feature_dim] 或 [M, feature_dim]

        返回：
            加密后的查询 [2*feature_dim] 或 [M, 2*feature_dim]
        """
        self._ensure_initialized()

        if isinstance(query, np.ndarray):
            query = torch.from_numpy(query)

        encrypted = self.aspe.encrypt_query(query)

        return encrypted.cpu().numpy()

    def compute_similarity(self,
                          encrypted_query: np.ndarray,
                          encrypted_db: np.ndarray) -> np.ndarray:
        """
        计算密文相似度（内积）。

        参数：
            encrypted_query: 加密查询 [M, 2*feature_dim]
            encrypted_db: 加密数据库 [N, 2*feature_dim]

        返回：
            相似度矩阵 [M, N]（越高越相似）
        """
        self._ensure_initialized()

        if isinstance(encrypted_query, np.ndarray):
            encrypted_query = torch.from_numpy(encrypted_query)
        if isinstance(encrypted_db, np.ndarray):
            encrypted_db = torch.from_numpy(encrypted_db)

        encrypted_query = encrypted_query.to(self.device)
        encrypted_db = encrypted_db.to(self.device)

        return self.aspe.ciphertext_inner_product(
            encrypted_db, encrypted_query
        ).cpu().numpy()

    def compute_distance(self,
                        encrypted_query: np.ndarray,
                        encrypted_db: np.ndarray) -> np.ndarray:
        """
        计算密文距离（负内积）。

        注意：CIR 使用内积作为相似度，距离为负相似度。

        参数：
            encrypted_query: 加密查询 [M, 2*feature_dim]
            encrypted_db: 加密数据库 [N, 2*feature_dim]

        返回：
            距离矩阵 [M, N]（越小越相似）
        """
        similarity = self.compute_similarity(encrypted_query, encrypted_db)
        return -similarity

    def get_status(self) -> Dict[str, Any]:
        """获取服务状态。"""
        return {
            "service_type": "cir_encryption",
            "feature_dim": self.feature_dim,
            "encrypted_dim": 2 * self.feature_dim,
            "device": str(self.device),
            "keys_loaded": self.aspe is not None and self.aspe.M1 is not None,
            "scheme": "ASPE (SkNN 风格)",
            "database_encrypted": self._encrypted_database is not None,
            "database_size": self._encrypted_database.shape[0] if self._encrypted_database is not None else 0
        }

    def get_dim(self) -> int:
        """获取特征维度。"""
        return self.feature_dim

    def get_encrypted_dim(self) -> int:
        """获取加密后维度。"""
        return 2 * self.feature_dim

    def has_keys(self) -> bool:
        """检查是否已有密钥。"""
        return self.aspe is not None and self.aspe.M1 is not None

    def verify_consistency(self,
                          plaintext_data: np.ndarray,
                          num_samples: int = 10) -> Dict[str, Any]:
        """
        验证加密前后内积一致性。

        参数：
            plaintext_data: 明文特征 [N, feature_dim]
            num_samples: 采样数量

        返回：
            包含验证结果的字典
        """
        self._ensure_initialized()

        if isinstance(plaintext_data, np.ndarray):
            plaintext_data = torch.from_numpy(plaintext_data)

        plaintext_data = plaintext_data.to(self.device)

        # 验证内积保持性
        result = self.aspe.verify_inner_product_preservation(
            db_features=plaintext_data,
            query_features=plaintext_data,
            num_samples=num_samples
        )

        return {
            "verified": result['passed'],
            "mean_error": result['mean_absolute_diff'],
            "max_error": result['max_absolute_diff'],
            "num_samples": result['num_samples']
        }


# 全局服务实例
_cir_encryption_services: Dict[int, CIREncryptionService] = {}


def get_cir_encryption_service(feature_dim: int = 2048, device: Optional[str] = None) -> CIREncryptionService:
    """
    获取 CIR 加密服务实例。

    参数：
        feature_dim: 特征维度
        device: 计算设备

    返回：
        CIREncryptionService 实例
    """
    if feature_dim not in _cir_encryption_services:
        _cir_encryption_services[feature_dim] = CIREncryptionService(
            feature_dim=feature_dim, device=device
        )
    return _cir_encryption_services[feature_dim]