"""
ASPE 方案的密钥生成

为 ASPE 加密方案生成加密密钥（可逆矩阵）。
"""

import numpy as np
from typing import Tuple
from config.aspe_config import RANDOM_SEED


class KeyGenerator:
    """ASPE 密钥生成器。"""

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

        参数：
            n: 矩阵维度

        返回：
            n×n 可逆矩阵
        """
        while True:
            M = np.random.randn(n, n)
            try:
                det = np.linalg.det(M)
                if abs(det) > 1e-6:
                    return M
            except np.linalg.LinAlgError:
                continue

    def generate_splitting_config(self, d: int, split_ratio: float = 0.5) -> np.ndarray:
        """
        生成随机拆分配置位向量。

        参数：
            d: 原始维度
            split_ratio: 要拆分的维度比例（默认 0.5）

        返回：
            d 维二进制向量 S
        """
        S = np.random.rand(d) < split_ratio
        return S.astype(int)


if __name__ == "__main__":
    # 测试密钥生成
    print("测试 ASPE 密钥生成")

    keygen = KeyGenerator()

    d = 64
    M = keygen.generate_invertible_matrix(d)
    print(f"矩阵 M 形状: {M.shape}")
    print(f"矩阵 M 行列式: {np.linalg.det(M):.6f}")

    S = keygen.generate_splitting_config(d)
    print(f"拆分配置 S: {S}")