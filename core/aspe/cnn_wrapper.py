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
    - 查询端：根据 S 向量，S=0 时拆分 (r=0 固定)，S=1 时复制
    - 使用 M1, M2 两个随机矩阵进行线性变换
    - 密文内积 = 明文内积（保持性）

    优化特性：
    - 预计算逆矩阵，避免重复计算
    - 条件数检查，警告病态矩阵
    - float64 高精度计算，减少累积误差
    - 向量化操作，提高批量处理效率
    - 查询端固定 r=0，最大化数值稳定性

    注意：
    - 特征维度 d 加密后变为 2d
    - 新版密钥文件包含预计算的逆矩阵（float64）
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

        # 预计算的逆矩阵（用于查询端加密）
        self.M1_inv: Optional[torch.Tensor] = None
        self.M2_inv: Optional[torch.Tensor] = None

        logger.info(f"ASPEForCNN 初始化：feature_dim={feature_dim}, device={self.device}")

    def generate_keys(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        生成 SkNN 密钥。

        包括：
        - 条件数检查，警告病态矩阵
        - 预计算逆矩阵（float64 高精度）

        返回：
            M1, M2, S 三个密钥
        """
        d = self.feature_dim

        # 生成两个 d×d 的随机矩阵
        self.M1 = torch.randn(d, d, device=self.device)
        self.M2 = torch.randn(d, d, device=self.device)

        # 生成 d 维二值向量 S
        self.S = torch.randint(0, 2, (d,), dtype=torch.float32, device=self.device)

        # 检查条件数（使用 float64 避免溢出）
        cond_M1 = torch.linalg.cond(self.M1.to(torch.float64))
        cond_M2 = torch.linalg.cond(self.M2.to(torch.float64))

        if cond_M1 > 1e10:
            logger.warning(f"M1 条件数过大 ({cond_M1:.2e})，可能影响数值稳定性")
        if cond_M2 > 1e10:
            logger.warning(f"M2 条件数过大 ({cond_M2:.2e})，可能影响数值稳定性")

        # 预计算逆矩阵（保持 float64 高精度，避免精度丢失）
        self.M1_inv = torch.linalg.inv(self.M1.to(torch.float64))
        self.M2_inv = torch.linalg.inv(self.M2.to(torch.float64))

        logger.info(f"密钥生成完成：M1.shape={self.M1.shape}, S 中 1 的比例={self.S.mean().item():.2%}")
        return self.M1, self.M2, self.S

    def load_keys(self, keys_path: Union[str, Path]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        从文件加载密钥。

        如果文件中包含预计算的逆矩阵则直接加载，
        否则重新计算逆矩阵。

        参数：
            keys_path: 密钥文件路径

        返回：
            M1, M2, S 三个密钥
        """
        keys = torch.load(keys_path, map_location=self.device, weights_only=False)
        self.M1 = keys['M1']
        self.M2 = keys['M2']
        self.S = keys['S']

        # 加载或计算逆矩阵
        if 'M1_inv' in keys and 'M2_inv' in keys:
            self.M1_inv = keys['M1_inv']
            self.M2_inv = keys['M2_inv']
            # 如果加载的逆矩阵是 float32，重新计算为 float64
            if self.M1_inv.dtype == torch.float32:
                logger.warning("检测到旧格式逆矩阵 (float32)，重新计算为 float64")
                self.M1_inv = torch.linalg.inv(self.M1.to(torch.float64))
                self.M2_inv = torch.linalg.inv(self.M2.to(torch.float64))
            logger.info(f"密钥加载完成（含逆矩阵）：{keys_path}")
        else:
            # 兼容旧格式：重新计算逆矩阵（保持 float64）
            self.M1_inv = torch.linalg.inv(self.M1.to(torch.float64))
            self.M2_inv = torch.linalg.inv(self.M2.to(torch.float64))
            logger.info(f"密钥加载完成（重新计算逆矩阵）：{keys_path}")

        return self.M1, self.M2, self.S

    def save_keys(self, save_path: Union[str, Path]):
        """
        保存密钥到文件。

        包括 M1, M2, S 以及预计算的逆矩阵 M1_inv, M2_inv。

        参数：
            save_path: 保存路径
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        keys_dict = {
            'M1': self.M1,
            'M2': self.M2,
            'S': self.S,
            'M1_inv': self.M1_inv,
            'M2_inv': self.M2_inv
        }
        torch.save(keys_dict, save_path)
        logger.info(f"密钥已保存：{save_path}")

    def _encrypt_db_feature(self, p: torch.Tensor) -> torch.Tensor:
        """
        加密数据库特征（建库端）。

        规则：
        - S[i]=0: p1[i]=p[i], p2[i]=p[i] (复制)
        - S[i]=1: p1[i]=r, p2[i]=p[i]-r (随机拆分)

        使用向量化操作和 float64 高精度计算。

        参数：
            p: 明文特征 [d]

        返回：
            密文特征 [2d]
        """
        d = len(p)
        device = p.device

        # 确保密钥在正确设备上
        M1 = self.M1.to(device)
        M2 = self.M2.to(device)
        S = self.S.to(device)

        # 向量化拆分
        r = torch.randn(d, device=device)
        S_bool = (S == 1)

        # S=0 时复制，S=1 时随机拆分
        p1 = torch.where(S_bool, r, p)
        p2 = torch.where(S_bool, p - r, p)

        # float64 高精度计算，结果转回 float32
        enc_part1 = torch.matmul(M1.to(torch.float64).T, p1.to(torch.float64))
        enc_part2 = torch.matmul(M2.to(torch.float64).T, p2.to(torch.float64))

        return torch.cat((enc_part1, enc_part2)).to(torch.float32)

    def _encrypt_query_feature(self, p: torch.Tensor) -> torch.Tensor:
        """
        加密查询特征（查询端/陷阱门）。

        规则（与建库端相反）：
        - S[i]=0: q1[i]=0, q2[i]=p[i] (拆分，r=0 固定，消除浮点误差)
        - S[i]=1: q1[i]=p[i], q2[i]=p[i] (复制)

        使用预计算逆矩阵和 float64 高精度计算。

        参数：
            p: 明文查询特征 [d]

        返回：
            密文陷阱门 [2d]
        """
        d = len(p)
        device = p.device

        # 确保密钥在正确设备上（逆矩阵已保持 float64）
        S = self.S.to(device)
        M1_inv = self.M1_inv.to(device)
        M2_inv = self.M2_inv.to(device)

        # 向量化拆分（S=0 时 r=0 固定，S=1 时复制）
        S_bool = (S == 0)

        # S=0: q1=0, q2=p (固定 r=0，数值稳定)
        # S=1: q1=p, q2=p (复制)
        q1 = torch.where(S_bool, torch.tensor(0.0, device=device), p)
        q2 = torch.where(S_bool, p, p)

        # 使用预计算逆矩阵（float64 高精度）
        trap_part1 = torch.matmul(M1_inv, q1.to(torch.float64))
        trap_part2 = torch.matmul(M2_inv, q2.to(torch.float64))

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
                                         db_features: torch.Tensor = None,
                                         query_features: torch.Tensor = None,
                                         num_samples: int = 10) -> Dict:
        """
        验证内积保持性。

        参数：
            db_features: 数据库特征 [N, d]，如果提供则使用真实数据
            query_features: 查询特征 [Q, d]，如果提供则使用真实数据
            num_samples: 测试样本数（仅在未提供真实数据时使用）

        返回：
            验证结果字典
        """
        if self.M1 is None:
            self.generate_keys()

        # 使用真实数据或生成随机数据
        use_real_data = db_features is not None and query_features is not None

        if use_real_data:
            # 确保是 torch.Tensor
            if isinstance(db_features, np.ndarray):
                db_features = torch.from_numpy(db_features)
            if isinstance(query_features, np.ndarray):
                query_features = torch.from_numpy(query_features)

            # 移动到同一设备
            device = self.device
            db_features = db_features.to(device)
            query_features = query_features.to(device)

            # 随机选择样本
            n_db = min(num_samples, len(db_features))
            n_query = min(num_samples, len(query_features))
            db_indices = torch.randperm(len(db_features))[:n_db]
            query_indices = torch.randperm(len(query_features))[:n_query]

            p_samples = db_features[db_indices]
            q_samples = query_features[query_indices]
        else:
            # 生成随机特征（移动到正确设备）
            p_samples = torch.randn(num_samples, self.feature_dim, device=self.device)
            q_samples = torch.randn(num_samples, self.feature_dim, device=self.device)

        # 计算多组样本的内积差异
        absolute_diffs = []
        relative_diffs = []

        for i in range(len(p_samples)):
            p = p_samples[i]
            q = q_samples[i]

            # 明文内积（在同一设备上计算）
            plaintext_ip = torch.dot(p, q)

            # 加密
            p_enc = self._encrypt_db_feature(p)
            q_enc = self._encrypt_query_feature(q)

            # 密文内积
            ciphertext_ip = torch.dot(p_enc, q_enc)

            # 计算差异
            diff = abs(plaintext_ip - ciphertext_ip)
            relative_diff = diff / (abs(plaintext_ip) + 1e-10)

            absolute_diffs.append(diff.item())
            relative_diffs.append(relative_diff.item())

        # 统计结果
        mean_abs_diff = np.mean(absolute_diffs)
        max_abs_diff = np.max(absolute_diffs)
        mean_rel_diff = np.mean(relative_diffs)
        max_rel_diff = np.max(relative_diffs)

        # 判断是否通过：
        # 1. 绝对误差 < 1e-3 (浮点精度范围内)
        # 2. 或相对误差 < 5% (考虑特征值可能较小的情况)
        passed = (mean_abs_diff < 1e-3) or (mean_rel_diff < 0.05)

        result = {
            'data_source': 'real' if use_real_data else 'random',
            'num_samples': len(p_samples),
            'mean_absolute_diff': float(mean_abs_diff),
            'max_absolute_diff': float(max_abs_diff),
            'mean_relative_diff': float(mean_rel_diff),
            'max_relative_diff': float(max_rel_diff),
            'absolute_diffs': absolute_diffs,
            'passed': bool(passed)
        }

        logger.info(f"内积保持性验证：{'通过' if result['passed'] else '失败'} "
                   f"(mean_abs={result['mean_absolute_diff']:.2e}, "
                   f"mean_rel={result['mean_relative_diff']:.4f}, "
                   f"data={'真实' if use_real_data else '随机'})")

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
