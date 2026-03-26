"""
ASPE 服务：非对称标量积保持加密

封装 ASPE 加密方案，提供：
- 检索库哈希码加密
- 查询陷阱门生成
- 密文汉明距离计算
- 密文 mAP 评估
- 密钥持久化支持

基于 SIGMOD'09 论文实现 ASPE 算法（SkNN 风格），密文内积 = 明文内积。
"""

import numpy as np
import torch
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import sys
import logging

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from core.aspe.dcmh_wrapper import ASPEForDCMH
from backend.app.services.key_manager import get_key_manager

logger = logging.getLogger(__name__)


class ASPEService:
    """
    ASPE 加密服务。

    基于 SIGMOD'09 论文实现 ASPE 算法（SkNN 风格），提供：
    - 初始化 ASPE 密钥（支持持久化）
    - 加密检索库哈希码
    - 生成查询陷阱门
    - 计算密文汉明距离
    - 计算密文 mAP

    特性：
    - 密文内积 = 明文内积
    - 双矩阵：M1, M2
    - 拆分向量：S
    """

    def __init__(self,
                 bit_dim: int = 64,
                 seed: int = 42,
                 key_path: Optional[str] = None,
                 auto_load_keys: bool = True):
        """
        初始化 ASPE 服务。

        参数：
            bit_dim: 哈希码位数（必须与 DCMH 一致）
            seed: ASPE 密钥生成的随机种子
            key_path: 可选的密钥保存/加载路径（已弃用，使用 KeyManager）
            auto_load_keys: 是否自动加载持久化密钥
        """
        self.bit_dim = bit_dim
        self.seed = seed

        # 初始化 ASPE 包装器
        self.aspe_wrapper = ASPEForDCMH(bit_dim=bit_dim, seed=seed)

        # 加密检索库缓存
        self.encrypted_database: Optional[np.ndarray] = None  # 加密的图像哈希码（用于标签→图像检索）
        self.encrypted_text_database: Optional[np.ndarray] = None  # 加密的文本哈希码（用于图像→标签检索）
        self.database_labels: Optional[np.ndarray] = None  # LAll 类别标签（用于 mAP 计算）
        self.database_codes: Optional[np.ndarray] = None

        # 自动加载持久化密钥
        if auto_load_keys:
            self._load_persistent_keys()

    def _load_persistent_keys(self):
        """从 KeyManager 加载持久化密钥。"""
        try:
            key_manager = get_key_manager()
            keys = key_manager.get_dcmh_keys(
                bit_dim=self.bit_dim,
                seed=self.seed
            )

            # 应用密钥到 ASPE 包装器
            if keys:
                if 'M1' in keys:
                    self.aspe_wrapper.M1 = keys['M1']
                if 'M2' in keys:
                    self.aspe_wrapper.M2 = keys['M2']
                if 'S' in keys:
                    self.aspe_wrapper.S = keys['S']
                if 'M1_inv' in keys and keys['M1_inv'] is not None:
                    self.aspe_wrapper.M1_inv = keys['M1_inv']
                if 'M2_inv' in keys and keys['M2_inv'] is not None:
                    self.aspe_wrapper.M2_inv = keys['M2_inv']
                logger.info(f"[ASPEService] 已加载持久化密钥")
        except Exception as e:
            logger.warning(f"[ASPEService] 加载持久化密钥失败：{e}，将使用临时密钥")

    def encrypt_database(self,
                        hash_codes: np.ndarray,
                        labels: Optional[np.ndarray] = None) -> np.ndarray:
        """
        加密图像哈希码数据库（用于标签→图像检索）。

        参数：
            hash_codes: {-1, +1}^{N×bit} 图像哈希码
            labels: LAll 类别标签 {0,1}^{N×L}（用于 mAP 计算，与加密无关）

        返回：
            {N×(2×bit)} 加密后的图像哈希码
        """
        # 加密哈希码
        encrypted = self.aspe_wrapper.GenEnc(hash_codes)

        # 缓存加密结果
        self.encrypted_database = encrypted
        self.database_codes = hash_codes
        if labels is not None:
            self.database_labels = labels

        return encrypted

    def encrypt_text_database(self,
                              text_hash_codes: np.ndarray) -> np.ndarray:
        """
        加密文本哈希码数据库（用于图像→标签检索）。

        参数：
            text_hash_codes: {-1, +1}^{N×bit} 文本哈希码

        返回：
            {N×(2×bit)} 加密后的文本哈希码
        """
        # 加密文本哈希码
        encrypted = self.aspe_wrapper.GenEnc(text_hash_codes)

        # 缓存加密结果
        self.encrypted_text_database = encrypted

        return encrypted

    def generate_trapdoor(self,
                         query_codes: np.ndarray) -> np.ndarray:
        """
        生成查询陷阱门。

        参数：
            query_codes: {-1, +1}^{M×bit} 查询哈希码

        返回：
            {M×(2×bit)} 加密查询（陷阱门）
        """
        return self.aspe_wrapper.GenTrap(query_codes)

    def compute_ciphertext_distances(self,
                                    encrypted_query: np.ndarray,
                                    encrypted_database: Optional[np.ndarray] = None) -> np.ndarray:
        """
        计算密文汉明距离。

        参数：
            encrypted_query: 加密查询 [M, 2×bit]
            encrypted_database: 加密检索库 [N, 2×bit]（可选，使用缓存）

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
                              query_labels: np.ndarray,
                              k: Optional[int] = None) -> float:
        """
        计算密文 mAP。

        参数：
            encrypted_query: 加密查询 [M, 2×bit]
            query_labels: 查询类别标签 LAll {0,1}^{M×L}
            k: 截断位置（默认使用全部）

        返回：
            mAP 值
        """
        if self.encrypted_database is None:
            raise ValueError("加密检索库未初始化")

        if self.database_labels is None:
            raise ValueError("检索库类别标签未设置")

        return self.aspe_wrapper.calc_ciphertext_map(
            encrypted_qB=encrypted_query,
            encrypted_rB=self.encrypted_database,
            query_L=query_labels,
            retrieval_L=self.database_labels,
            k=k
        )

    def compute_plaintext_map(self,
                             query_codes: np.ndarray,
                             query_labels: np.ndarray,
                             k: Optional[int] = None) -> float:
        """
        计算明文 mAP（用于对比）。

        参数：
            query_codes: {-1, +1}^{M×bit} 查询哈希码
            query_labels: 查询类别标签 LAll {0,1}^{M×L}
            k: 截断位置

        返回：
            mAP 值
        """
        if self.database_codes is None:
            raise ValueError("检索库哈希码未设置")

        if self.database_labels is None:
            raise ValueError("检索库类别标签未设置")

        # 计算明文汉明距离
        distances = self._plaintext_hamming_distance(query_codes, self.database_codes)

        # 计算 mAP
        return self._compute_map_from_distances(
            distances, query_labels, self.database_labels, k
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
                                   query_labels: np.ndarray,
                                   retrieval_labels: np.ndarray,
                                   k: Optional[int] = None) -> float:
        """从距离矩阵计算 mAP。"""
        num_query = distances.shape[0]
        map_score = 0.0

        if k is None:
            k = retrieval_labels.shape[0]

        for i in range(num_query):
            q_L = query_labels[i:i+1]

            # 计算相关性
            gnd = (q_L @ retrieval_labels.T > 0).squeeze().astype(np.float32)

            tsum = np.sum(gnd)
            if tsum == 0:
                continue

            # 使用统一的排序逻辑：四舍五入 + lexsort 确保相同距离时的确定性排序
            dist_i = distances[i]
            dist_rounded = np.round(dist_i, decimals=10)
            ind = np.lexsort((np.arange(len(dist_rounded)), dist_rounded))
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
                          query_labels: np.ndarray,
                          num_samples: int = 10) -> Dict[str, float]:
        """
        验证 ASPE 加密前后 mAP 一致性。

        参数：
            query_codes: 查询哈希码
            query_labels: 查询类别标签 LAll
            num_samples: 采样查询数量

        返回：
            包含明文 mAP、密文 mAP、差异的字典
        """
        # 加密查询
        encrypted_query = self.generate_trapdoor(query_codes[:num_samples])

        # 计算明文 mAP
        plaintext_map = self.compute_plaintext_map(
            query_codes[:num_samples], query_labels[:num_samples]
        )

        # 计算密文 mAP
        ciphertext_map = self.compute_ciphertext_map(
            encrypted_query, query_labels[:num_samples]
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
            "ciphertext_dim": 2 * self.bit_dim,
            "seed": self.seed,
            "scheme": "ASPE (SkNN 风格)",
            "database_encrypted": self.encrypted_database is not None,
            "database_size": self.encrypted_database.shape[0] if self.encrypted_database is not None else 0,
            "database_labels_set": self.database_labels is not None
        }


# 全局服务实例
_aspe_service: Optional[ASPEService] = None


def get_aspe_service(**kwargs) -> ASPEService:
    """获取或创建 ASPE 服务单例。"""
    global _aspe_service
    if _aspe_service is None:
        _aspe_service = ASPEService(**kwargs)
    return _aspe_service