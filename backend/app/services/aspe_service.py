"""
ASPE 服务：非对称标量积保持加密

封装 ASPE 加密方案，提供：
- 检索库哈希码加密
- 查询陷阱门生成
- 密文汉明距离计算
- 密文 mAP 评估
- 密钥持久化支持

使用 Scheme 2（双矩阵增强方案）提供更高安全级别。
"""

import numpy as np
import torch
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import sys
import logging

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from core.aspe.dcmh_wrapper_v2 import ASPEForDCMHv2
from core.aspe.scheme2 import ASPEScheme2
from backend.app.services.key_manager import get_key_manager

logger = logging.getLogger(__name__)


class ASPEService:
    """
    ASPE 加密服务。

    使用 Scheme 2（双矩阵增强方案）提供：
    - 初始化 ASPE 密钥（支持持久化）
    - 加密检索库哈希码
    - 生成查询陷阱门
    - 计算密文汉明距离
    - 计算密文 mAP

    安全特性：
    - 安全级别：3（抵抗已知明文攻击）
    - 双矩阵：M1, M2
    - 拆分向量：S
    """

    def __init__(self,
                 bit_dim: int = 64,
                 d_prime: Optional[int] = None,
                 seed: int = 42,
                 key_path: Optional[str] = None,
                 auto_load_keys: bool = True):
        """
        初始化 ASPE 服务。

        参数：
            bit_dim: 哈希码位数（必须与 DCMH 一致）
            d_prime: 扩展维度（默认：max(bit_dim+1, 80)）
            seed: ASPE 密钥生成的随机种子
            key_path: 可选的密钥保存/加载路径（已弃用，使用 KeyManager）
            auto_load_keys: 是否自动加载持久化密钥
        """
        self.bit_dim = bit_dim
        self.d_prime = d_prime
        self.seed = seed

        # 初始化 ASPE 包装器 V2 (Scheme 2)
        self.aspe_wrapper = ASPEForDCMHv2(bit_dim=bit_dim, d_prime=d_prime, seed=seed)

        # 加密检索库缓存
        self.encrypted_database: Optional[np.ndarray] = None
        self.database_tags: Optional[np.ndarray] = None
        self.database_codes: Optional[np.ndarray] = None

        # 自动加载持久化密钥
        if auto_load_keys:
            self._load_persistent_keys()

    def _load_persistent_keys(self):
        """从 KeyManager 加载持久化密钥。"""
        try:
            key_manager = get_key_manager()
            keys = key_manager.get_dcmh_v2_keys(
                bit_dim=self.bit_dim,
                d_prime=self.d_prime,
                seed=self.seed
            )

            # 应用密钥到 ASPE 包装器
            # Scheme 2 使用 M1, M2, S, w 矩阵
            if keys:
                if 'M1' in keys:
                    self.aspe_wrapper.aspe.M1 = keys['M1']
                if 'M2' in keys:
                    self.aspe_wrapper.aspe.M2 = keys['M2']
                if 'S' in keys:
                    self.aspe_wrapper.aspe.S = keys['S']
                if 'w' in keys:
                    self.aspe_wrapper.aspe.w = keys['w']
                if 'M1_inv' in keys and keys['M1_inv'] is not None:
                    self.aspe_wrapper.aspe.M1_inv = keys['M1_inv']
                if 'M2_inv' in keys and keys['M2_inv'] is not None:
                    self.aspe_wrapper.aspe.M2_inv = keys['M2_inv']
                if 'd_prime' in keys:
                    self.aspe_wrapper.actual_d_prime = keys['d_prime']
                    self.aspe_wrapper.aspe.d_prime = keys['d_prime']
                logger.info(f"[ASPEService] 已加载 Scheme 2 持久化密钥")
        except Exception as e:
            logger.warning(f"[ASPEService] 加载持久化密钥失败：{e}，将使用临时密钥")

    def encrypt_database(self,
                        hash_codes: np.ndarray,
                        tags: Optional[np.ndarray] = None) -> np.ndarray:
        """
        加密检索库哈希码。

        参数：
            hash_codes: {-1, +1}^{N×bit} 检索库哈希码
            tags: 可选的标签（用于 mAP 计算）

        返回：
            {N×(bit+1)} 加密检索库
        """
        # 加密哈希码
        encrypted = self.aspe_wrapper.GenEnc(hash_codes)

        # 缓存加密结果
        self.encrypted_database = encrypted
        self.database_codes = hash_codes
        if tags is not None:
            self.database_tags = tags

        return encrypted

    def generate_trapdoor(self,
                         query_codes: np.ndarray,
                         r: Optional[float] = None) -> np.ndarray:
        """
        生成查询陷阱门。

        参数：
            query_codes: {-1, +1}^{M×bit} 查询哈希码
            r: 可选的缩放因子

        返回：
            {M×(bit+1)} 加密查询（陷阱门）
        """
        return self.aspe_wrapper.GenTrap(query_codes, r)

    def compute_ciphertext_distances(self,
                                    encrypted_query: np.ndarray,
                                    encrypted_database: Optional[np.ndarray] = None) -> np.ndarray:
        """
        计算密文汉明距离。

        参数：
            encrypted_query: 加密查询 [M, bit+1]
            encrypted_database: 加密检索库 [N, bit+1]（可选，使用缓存）

        返回：
            [M, N] 汉明距离矩阵
        """
        if encrypted_database is None:
            if self.encrypted_database is None:
                raise ValueError("未提供加密检索库且无缓存")
            encrypted_database = self.encrypted_database

        return self.aspe_wrapper.ciphertext_hamming_distance(
            encrypted_query, encrypted_database
        )

    def compute_ciphertext_map(self,
                              encrypted_query: np.ndarray,
                              query_tags: np.ndarray,
                              k: Optional[int] = None) -> float:
        """
        计算密文 mAP。

        参数：
            encrypted_query: 加密查询 [M, bit+1]
            query_tags: 查询标签 {0,1}^{M×L}
            k: 截断位置（默认使用全部）

        返回：
            mAP 值
        """
        if self.encrypted_database is None:
            raise ValueError("加密检索库未初始化")

        if self.database_tags is None:
            raise ValueError("检索库标签未设置")

        return self.aspe_wrapper.calc_ciphertext_map(
            encrypted_qB=encrypted_query,
            encrypted_rB=self.encrypted_database,
            query_L=query_tags,
            retrieval_L=self.database_tags,
            k=k
        )

    def compute_plaintext_map(self,
                             query_codes: np.ndarray,
                             query_tags: np.ndarray,
                             k: Optional[int] = None) -> float:
        """
        计算明文 mAP（用于对比）。

        参数：
            query_codes: {-1, +1}^{M×bit} 查询哈希码
            query_tags: 查询标签 {0,1}^{M×L}
            k: 截断位置

        返回：
            mAP 值
        """
        if self.database_codes is None:
            raise ValueError("检索库哈希码未设置")

        if self.database_tags is None:
            raise ValueError("检索库标签未设置")

        # 计算明文汉明距离
        distances = self._plaintext_hamming_distance(query_codes, self.database_codes)

        # 计算 mAP
        return self._compute_map_from_distances(
            distances, query_tags, self.database_tags, k
        )

    def _plaintext_hamming_distance(self,
                                   codes1: np.ndarray,
                                   codes2: np.ndarray) -> np.ndarray:
        """计算两个哈希码集合之间的汉明距离。"""
        # 汉明距离 = 0.5 × (bit - 内积)
        inner_products = np.dot(codes1, codes2.T)
        distances = 0.5 * (self.bit_dim - inner_products)
        return distances

    def _compute_map_from_distances(self,
                                   distances: np.ndarray,
                                   query_tags: np.ndarray,
                                   retrieval_tags: np.ndarray,
                                   k: Optional[int] = None) -> float:
        """从距离矩阵计算 mAP。"""
        num_query = distances.shape[0]
        map_score = 0.0

        if k is None:
            k = retrieval_tags.shape[0]

        for i in range(num_query):
            q_L = query_tags[i:i+1]

            # 计算相关性
            gnd = (q_L @ retrieval_tags.T > 0).squeeze().astype(np.float32)

            tsum = np.sum(gnd)
            if tsum == 0:
                continue

            # 按距离排序（升序）
            ind = np.argsort(distances[i])
            gnd_sorted = gnd[ind]

            # 计算 AP
            total = min(k, int(tsum))
            count = np.arange(1, total + 1, dtype=np.float32)
            tindex = np.flatnonzero(gnd_sorted)[:total].astype(np.float32) + 1.0

            ap = np.mean(count / tindex)
            map_score += ap

        map_score /= num_query
        return map_score

    def verify_consistency(self,
                          query_codes: np.ndarray,
                          query_tags: np.ndarray,
                          num_samples: int = 10) -> Dict[str, float]:
        """
        验证 ASPE 加密前后 mAP 一致性。

        参数：
            query_codes: 查询哈希码
            query_tags: 查询标签
            num_samples: 采样查询数量

        返回：
            包含明文 mAP、密文 mAP、差异的字典
        """
        # 加密查询
        encrypted_query = self.generate_trapdoor(query_codes[:num_samples])

        # 计算明文 mAP
        plaintext_map = self.compute_plaintext_map(
            query_codes[:num_samples], query_tags[:num_samples]
        )

        # 计算密文 mAP
        ciphertext_map = self.compute_ciphertext_map(
            encrypted_query, query_tags[:num_samples]
        )

        return {
            "plaintext_map": plaintext_map,
            "ciphertext_map": ciphertext_map,
            "difference": abs(plaintext_map - ciphertext_map),
            "consistent": abs(plaintext_map - ciphertext_map) < 1e-3
        }

    def get_status(self) -> Dict[str, Any]:
        """获取服务状态。"""
        return {
            "bit_dim": self.bit_dim,
            "d_prime": self.aspe_wrapper.actual_d_prime,
            "ciphertext_dim": 2 * self.aspe_wrapper.actual_d_prime,
            "seed": self.seed,
            "scheme": "Scheme 2 (双矩阵增强方案)",
            "security_level": 3,
            "database_encrypted": self.encrypted_database is not None,
            "database_size": self.encrypted_database.shape[0] if self.encrypted_database is not None else 0,
            "database_tags_set": self.database_tags is not None
        }


# 全局服务实例
_aspe_service: Optional[ASPEService] = None


def get_aspe_service(**kwargs) -> ASPEService:
    """获取或创建 ASPE 服务单例。"""
    global _aspe_service
    if _aspe_service is None:
        _aspe_service = ASPEService(**kwargs)
    return _aspe_service