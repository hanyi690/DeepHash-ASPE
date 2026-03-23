"""
ASPE for CNN：CNN 图像检索的 ASPE 加密包装器

提供与 CNN 特征向量兼容的 ASPE 加密接口，支持：
- SkNN 双矩阵方案
- 数据库特征加密
- 查询陷阱门生成
- 密文相似度搜索
"""

import torch
import numpy as np
from typing import Tuple, Optional, List, Dict, Union
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ASPEForCNN:
    """
    CNN 特征向量的 ASPE 加密包装器。

    实现 SkNN (Secure k-Nearest Neighbors) 方案，
    基于 ASPE (Asymmetric Scalar Product-Preserving Encryption) 原理。

    加密原理：
    - 建库端：根据 S 向量，S=0 时复制，S=1 时随机拆分
    - 查询端：根据 S 向量，S=0 时拆分 (r=0)，S=1 时复制
    - 使用 M1, M2 两个随机矩阵进行线性变换
    - 密文内积 = 明文内积（保持性）

    注意：
    - 本实现使用 Float64 计算逆矩阵以消除浮点误差
    - 特征维度 d 加密后变为 2d
    """

    def __init__(self,
                 feature_dim: int = 2048,
                 seed: Optional[int] = None,
                 device: Optional[str] = None):
        """
        初始化 ASPE for CNN。

        参数：
            feature_dim: 特征维度（默认 2048，对应 ResNet101-GeM）
            seed: 随机种子（可选，用于可复现）
            device: 计算设备
        """
        self.feature_dim = feature_dim
        self.device = torch.device(device) if device else torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )

        # 设置随机种子（可选）
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        # 密钥（需要生成或加载）
        self.M1: Optional[torch.Tensor] = None
        self.M2: Optional[torch.Tensor] = None
        self.S: Optional[torch.Tensor] = None

        logger.info(f"ASPEForCNN 初始化：feature_dim={feature_dim}, device={self.device}")

    def generate_keys(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        生成 SkNN 密钥。

        返回：
            M1, M2, S 三个密钥
        """
        d = self.feature_dim

        # 生成两个 d×d 的随机矩阵
        self.M1 = torch.randn(d, d, device=self.device)
        self.M2 = torch.randn(d, d, device=self.device)

        # 生成 d 维二值向量 S
        self.S = torch.randint(0, 2, (d,), dtype=torch.float32, device=self.device)

        logger.info(f"密钥生成完成：M1.shape={self.M1.shape}, S 中 1 的比例={self.S.mean().item():.2%}")
        return self.M1, self.M2, self.S

    def load_keys(self, keys_path: Union[str, Path]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """从文件加载密钥。"""
        keys = torch.load(keys_path, map_location=self.device, weights_only=False)
        self.M1 = keys['M1']
        self.M2 = keys['M2']
        self.S = keys['S']
        logger.info(f"密钥加载完成：{keys_path}")
        return self.M1, self.M2, self.S

    def save_keys(self, save_path: Union[str, Path]):
        """保存密钥到文件。"""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({'M1': self.M1, 'M2': self.M2, 'S': self.S}, save_path)
        logger.info(f"密钥已保存：{save_path}")

    def _encrypt_db_feature(self, p: torch.Tensor) -> torch.Tensor:
        """
        加密数据库特征（建库端）。

        规则：
        - S[i]=0: p1[i]=p[i], p2[i]=p[i] (复制)
        - S[i]=1: p1[i]=r, p2[i]=p[i]-r (随机拆分)

        参数：
            p: 明文特征 [d]

        返回：
            密文特征 [2d]
        """
        d = len(p)
        # 确保密钥在正确设备上
        M1 = self.M1.to(p.device)
        M2 = self.M2.to(p.device)
        S = self.S.to(p.device)

        p1 = torch.zeros(d, device=p.device)
        p2 = torch.zeros(d, device=p.device)

        for i in range(d):
            if S[i] == 0:
                p1[i] = p[i]
                p2[i] = p[i]
            else:
                r = torch.randn(1, device=p.device).item()
                p1[i] = r
                p2[i] = p[i] - r

        # 线性变换
        enc_part1 = torch.matmul(M1.T, p1)
        enc_part2 = torch.matmul(M2.T, p2)

        return torch.cat((enc_part1, enc_part2))

    def _encrypt_query_feature(self, p: torch.Tensor) -> torch.Tensor:
        """
        加密查询特征（查询端/陷阱门）。

        规则（与建库端相反）：
        - S[i]=0: q1[i]=0, q2[i]=p[i] (拆分，r=0)
        - S[i]=1: q1[i]=p[i], q2[i]=p[i] (复制)

        参数：
            p: 明文查询特征 [d]

        返回：
            密文陷阱门 [2d]
        """
        d = len(p)
        # 确保密钥在正确设备上
        M1 = self.M1.to(p.device)
        M2 = self.M2.to(p.device)
        S = self.S.to(p.device)

        q1 = torch.zeros(d, device=p.device)
        q2 = torch.zeros(d, device=p.device)

        for i in range(d):
            if S[i] == 0:
                # r=0 消除浮点误差
                q1[i] = 0.0
                q2[i] = p[i]
            else:
                q1[i] = p[i]
                q2[i] = p[i]

        # Float64 高精度计算
        M1_double = M1.to(torch.float64)
        M2_double = M2.to(torch.float64)
        q1_double = q1.to(torch.float64)
        q2_double = q2.to(torch.float64)

        # 逆变换
        trap_part1 = torch.matmul(torch.linalg.inv(M1_double), q1_double)
        trap_part2 = torch.matmul(torch.linalg.inv(M2_double), q2_double)

        return torch.cat((trap_part1, trap_part2)).to(torch.float32)

    def encrypt_database(self,
                        features: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """
        批量加密数据库特征。

        参数：
            features: [N, d] 明文特征矩阵

        返回：
            [N, 2d] 密文特征矩阵
        """
        if isinstance(features, np.ndarray):
            features = torch.from_numpy(features)

        if features.dim() == 1:
            features = features.unsqueeze(0)

        n, d = features.shape
        if d != self.feature_dim:
            raise ValueError(f"特征维度不匹配：输入{d}, 期望{self.feature_dim}")

        encrypted = []
        for i in range(n):
            enc_feat = self._encrypt_db_feature(features[i])
            encrypted.append(enc_feat)

        return torch.stack(encrypted)

    def encrypt_query(self,
                     query_feature: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """
        加密查询特征（生成陷阱门）。

        参数：
            query_feature: [d] 或 [1, d] 查询特征

        返回：
            [2d] 密文陷阱门
        """
        if isinstance(query_feature, np.ndarray):
            query_feature = torch.from_numpy(query_feature)

        if query_feature.dim() == 2:
            query_feature = query_feature.squeeze(0)

        return self._encrypt_query_feature(query_feature)

    def ciphertext_inner_product(self,
                                encrypted_db: torch.Tensor,
                                encrypted_query: torch.Tensor) -> torch.Tensor:
        """
        计算密文内积（等价于明文内积）。

        参数：
            encrypted_db: [N, 2d] 加密数据库
            encrypted_query: [2d] 加密查询

        返回：
            [N] 内积分数
        """
        if encrypted_query.dim() == 1:
            encrypted_query = encrypted_query.unsqueeze(0)

        return torch.matmul(encrypted_db, encrypted_query.t()).squeeze()

    def search(self,
               encrypted_db: torch.Tensor,
               query_feature: Union[torch.Tensor, np.ndarray],
               top_k: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        在加密数据库中搜索最近邻。

        参数：
            encrypted_db: [N, 2d] 加密数据库
            query_feature: [d] 查询特征
            top_k: 返回结果数量

        返回：
            (scores, indices) Top-K 分数和索引
        """
        # 加密查询
        enc_query = self.encrypt_query(query_feature)

        # 计算密文内积
        scores = self.ciphertext_inner_product(encrypted_db, enc_query)

        # Top-K
        topk_scores, topk_indices = torch.topk(scores, k=top_k)

        return topk_scores, topk_indices

    def verify_inner_product_preservation(self,
                                         num_samples: int = 10) -> Dict:
        """
        验证内积保持性。

        参数：
            num_samples: 测试样本数

        返回：
            验证结果字典
        """
        if self.M1 is None:
            self.generate_keys()

        # 生成随机特征
        p = torch.randn(self.feature_dim)
        q = torch.randn(self.feature_dim)

        # 明文内积
        plaintext_ip = torch.dot(p, q)

        # 加密
        p_enc = self._encrypt_db_feature(p)
        q_enc = self._encrypt_query_feature(q)

        # 密文内积
        ciphertext_ip = torch.dot(p_enc, q_enc)

        # 计算差异
        diff = abs(plaintext_ip - ciphertext_ip)
        relative_diff = diff / (abs(plaintext_ip) + 1e-10)

        result = {
            'plaintext_ip': plaintext_ip.item(),
            'ciphertext_ip': ciphertext_ip.item(),
            'absolute_diff': diff.item(),
            'relative_diff': relative_diff.item(),
            'passed': relative_diff.item() < 1e-3
        }

        logger.info(f"内积保持性验证：{'通过' if result['passed'] else '失败'} "
                   f"(diff={result['absolute_diff']:.2e})")

        return result


# 便捷函数
def create_cnn_aspe(feature_dim: int = 2048,
                   seed: Optional[int] = None,
                   keys_path: Optional[str] = None,
                   device: Optional[str] = None) -> ASPEForCNN:
    """
    创建并初始化 ASPEForCNN 实例。

    参数：
        feature_dim: 特征维度
        seed: 随机种子
        keys_path: 密钥文件路径（如果提供则加载，否则生成新密钥）
        device: 计算设备

    返回：
        初始化好的 ASPEForCNN 实例
    """
    aspe = ASPEForCNN(feature_dim=feature_dim, device=device)

    if keys_path:
        aspe.load_keys(keys_path)
    else:
        aspe.generate_keys()

    return aspe
