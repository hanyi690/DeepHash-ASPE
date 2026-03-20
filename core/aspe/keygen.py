"""
ASPE 方案的密钥生成

为 ASPE 方案 1 和方案 2 生成加密密钥（可逆矩阵）。
"""

import numpy as np
from typing import Tuple
from config.aspe_config import RANDOM_SEED


class KeyGenerator:
    """ASPE 密钥生成的基类。"""

    def __init__(self, seed: int = RANDOM_SEED):
        """
        初始化密钥生成器。

        参数：
            seed: 用于可重复性的随机种子
        """
        self.seed = seed
        np.random.seed(seed)

    def generate_invertible_matrix(self, n: int) -> np.ndarray:
        """
        生成 n×n 可逆矩阵。

        策略：
        1. 生成随机矩阵
        2. 通过检查行列式确保可逆性
        3. 如果是奇异矩阵，重新生成

        参数：
            n: 矩阵维度

        返回：
            n×n 可逆矩阵
        """
        while True:
            # 从标准正态分布生成随机矩阵
            M = np.random.randn(n, n)

            # 检查矩阵是否可逆
            try:
                # 计算行列式
                det = np.linalg.det(M)
                if abs(det) > 1e-6:  # 非奇异
                    return M
            except np.linalg.LinAlgError:
                # 矩阵是奇异矩阵，重试
                continue

    def generate_splitting_config(self, d: int, split_ratio: float = 0.5) -> np.ndarray:
        """
        为方案 2 生成随机拆分配置位向量。

        S[i] = 1 表示维度 i 在两个份额之间拆分
        S[i] = 0 表示维度 i 在两个份额中保持完整

        参数：
            d: 原始维度
            split_ratio: 要拆分的维度比例（默认 0.5）

        返回：
            d 维二进制向量 S
        """
        S = np.random.rand(d) < split_ratio
        return S.astype(int)

    def generate_random_splits(self, value: float) -> Tuple[float, float]:
        """
        将值拆分为两个和为原始值的随机份额。

        参数：
            value: 要拆分的值

        返回：
            (share_a, share_b) 元组，使得 share_a + share_b = value
        """
        share_a = np.random.randn() * value
        share_b = value - share_a
        return share_a, share_b


def generate_key_scheme1(d: int, seed: int = RANDOM_SEED) -> np.ndarray:
    """
    为方案 1 生成 (d+1)×(d+1) 可逆矩阵 M。

    方案 1 使用单个可逆矩阵 M 进行数据库和查询加密
   （分别使用 M 和 M^(-1)）。

    参数：
        d: 特征维度
        seed: 用于可重复性的随机种子

    返回：
        (d+1)×(d+1) 可逆矩阵 M
    """
    keygen = KeyGenerator(seed)
    M = keygen.generate_invertible_matrix(d + 1)
    return M


def generate_key_scheme2(d: int,
                         d_prime: int = None,
                         seed: int = RANDOM_SEED) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    为 ASPE 方案 2 生成密钥。

    方案 2 使用：
    - 两个可逆矩阵 M1, M2（d'×d'）
    - 拆分配置 S（d 位向量）
    - 人工维度的随机值 w

    参数：
        d: 原始特征维度
        d_prime: 扩展维度（默认为 max(d+1, 80) 以确保安全）
        seed: 用于可重复性的随机种子

    返回：
        (M1, M2, S, w) 元组：
        - M1: d'×d' 可逆矩阵
        - M2: d'×d' 可逆矩阵
        - S: d 位配置位向量
        - w: 人工维度的随机数
    """
    # 设置 d_prime 并满足安全约束
    if d_prime is None:
        d_prime = max(d + 1, 80)

    keygen = KeyGenerator(seed)

    # 生成两个可逆矩阵
    M1 = keygen.generate_invertible_matrix(d_prime)
    M2 = keygen.generate_invertible_matrix(d_prime)

    # 生成拆分配置
    S = keygen.generate_splitting_config(d)

    # 为人工维度生成随机值
    # 人工维度是从 (d+1) 到 d_prime
    num_artificial = d_prime - (d + 1)
    w = np.random.randn(num_artificial)

    return M1, M2, S, w


if __name__ == "__main__":
    # 测试密钥生成
    print("测试 ASPE 密钥生成")

    # 测试方案 1
    print("\n=== 方案 1 密钥生成 ===")
    d = 10
    M = generate_key_scheme1(d)
    print(f"矩阵 M 形状: {M.shape}")
    print(f"矩阵 M 行列式: {np.linalg.det(M):.6f}")

    # 测试方案 2
    print("\n=== 方案 2 密钥生成 ===")
    M1, M2, S, w = generate_key_scheme2(d)
    print(f"矩阵 M1 形状: {M1.shape}")
    print(f"矩阵 M2 形状: {M2.shape}")
    print(f"拆分配置 S: {S}")
    print(f"人工维度 w: {w}")
    print(f"M1 行列式: {np.linalg.det(M1):.6f}")
    print(f"M2 行列式: {np.linalg.det(M2):.6f}")
