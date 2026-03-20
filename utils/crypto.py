"""
加密工具

用于加密操作和安全评估的工具函数。
"""

import numpy as np
from typing import Tuple, List
import hashlib


def hash_vector(vector: np.ndarray, algorithm: str = 'sha256') -> str:
    """
    计算向量的加密哈希。

    Args:
        vector: 输入向量
        algorithm: 哈希算法('sha256', 'md5'等)

    Returns:
        十六进制哈希字符串
    """
    # 将向量转换为字节
    vector_bytes = vector.tobytes()

    # 计算哈希
    if algorithm == 'sha256':
        hash_obj = hashlib.sha256(vector_bytes)
    elif algorithm == 'md5':
        hash_obj = hashlib.md5(vector_bytes)
    elif algorithm == 'sha1':
        hash_obj = hashlib.sha1(vector_bytes)
    else:
        raise ValueError(f"Unknown hash algorithm: {algorithm}")

    return hash_obj.hexdigest()


def generate_random_vector(d: int, seed: int = None) -> np.ndarray:
    """
    生成随机d维向量。

    Args:
        d: 维度
        seed: 随机种子

    Returns:
        随机向量
    """
    if seed is not None:
        np.random.seed(seed)
    return np.random.randn(d)


def generate_random_vectors(n: int, d: int, seed: int = None) -> np.ndarray:
    """
    生成n个随机d维向量。

    Args:
        n: 向量数量
        d: 维度
        seed: 随机种子

    Returns:
        [n, d]随机向量数组
    """
    if seed is not None:
        np.random.seed(seed)
    return np.random.randn(n, d)


def compute_entropy(data: np.ndarray, bins: int = 50) -> float:
    """
    计算数据的香农熵。

    Args:
        data: 输入数据
        bins: 直方图的箱数

    Returns:
        熵值
    """
    # 计算直方图
    hist, _ = np.histogram(data, bins=bins, density=True)

    # 移除零值条目
    hist = hist[hist > 0]

    # 计算熵
    entropy = -np.sum(hist * np.log2(hist + 1e-10))

    return entropy


def verify_scalar_product_preservation(p_enc: np.ndarray,
                                       q_enc: np.ndarray,
                                       p: np.ndarray,
                                       q: np.ndarray,
                                       tolerance: float = 1e-6) -> bool:
    """
    验证ASPE是否保持内积结构。

    对于方案1: p_enc · q_enc应该与p · q成正比
    对于方案2: sum(p_enc · q_enc)应该与p · q成正比

    Args:
        p_enc: 加密的数据库点
        q_enc: 加密的查询点
        p: 原始数据库点
        q: 原始查询点
        tolerance: 数值误差的容差

    Returns:
        如果内积被保留(考虑缩放)则返回True
    """
    # 计算明文内积
    plaintext_ip = np.dot(p, q)

    # 计算密文内积
    if isinstance(p_enc, tuple):
        # 方案2: 两个分片
        ciphertext_ip = np.dot(p_enc[0], q_enc[0]) + np.dot(p_enc[1], q_enc[1])
    else:
        # 方案1: 单一向量
        ciphertext_ip = np.dot(p_enc, q_enc)

    # 检查比例性(允许随机缩放因子r)
    if abs(plaintext_ip) < tolerance:
        return abs(ciphertext_ip) < tolerance

    ratio = ciphertext_ip / plaintext_ip

    # 比率应该为正(缩放因子r > 0)
    return ratio > 0


def statistical_test_encryption(encrypted_data: np.ndarray,
                                original_data: np.ndarray) -> dict:
    """
    对加密数据进行统计测试。

    测试:
    - 均值和标准差比较
    - 原始数据和加密数据之间的相关性
    - 加密数据的熵

    Args:
        encrypted_data: 加密向量
        original_data: 原始向量

    Returns:
        测试结果字典
    """
    results = {}

    # 展平以进行统计测试
    enc_flat = encrypted_data.flatten()
    orig_flat = original_data.flatten()

    # 基本统计
    results['encrypted_mean'] = np.mean(enc_flat)
    results['encrypted_std'] = np.std(enc_flat)
    results['original_mean'] = np.mean(orig_flat)
    results['original_std'] = np.std(orig_flat)

    # 相关性
    if enc_flat.shape == orig_flat.shape:
        results['correlation'] = np.corrcoef(enc_flat, orig_flat)[0, 1]
    else:
        # 计算每个向量的相关性
        min_len = min(len(encrypted_data), len(original_data))
        correlations = []
        for i in range(min_len):
            if encrypted_data[i].shape == original_data[i].shape:
                corr = np.corrcoef(encrypted_data[i].flatten(),
                                  original_data[i].flatten())[0, 1]
                correlations.append(corr)
        results['correlation'] = np.mean(correlations) if correlations else 0.0

    # 熵
    results['encrypted_entropy'] = compute_entropy(enc_flat)
    results['original_entropy'] = compute_entropy(orig_flat)

    return results


def differential_attack_resistance(encrypted_db: np.ndarray,
                                   known_indices: List[int],
                                   known_vectors: np.ndarray) -> float:
    """
    测试对差分攻击的抵抗性(已知样本)。

    当某些样本已知时,测量未知样本泄露了多少信息。

    Args:
        encrypted_db: 完整的加密数据库
        known_indices: 已知样本的索引
        known_vectors: 已知索引处的原始向量

    Returns:
        泄露评分(越低越好)
    """
    # 这是一个简化测试
    # 实际中会使用更复杂的攻击方法

    # 计算已知加密与已知原始的统计
    known_encrypted = encrypted_db[known_indices]

    # 重建与实际之间的均方误差
    mse = np.mean((known_encrypted - known_vectors) ** 2)

    # 较低的MSE表示更多潜在泄露
    # 较高的MSE表示更好的安全性
    return 1.0 / (1.0 + mse)


def format_bytes(size: int) -> str:
    """
    将字节大小格式化为可读字符串。

    Args:
        size: 字节大小

    Returns:
        格式化字符串
    """
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} TB"


def estimate_storage_size(num_vectors: int,
                          dimension: int,
                          bytes_per_float: int = 8) -> str:
    """
    估计向量的存储大小。

    Args:
        num_vectors: 向量数量
        dimension: 向量维度
        bytes_per_float: 每个浮点数的字节数(默认8表示双精度)

    Returns:
        格式化的存储大小字符串
    """
    total_bytes = num_vectors * dimension * bytes_per_float
    return format_bytes(total_bytes)


if __name__ == "__main__":
    # 测试加密工具
    print("Testing Cryptographic Utilities")

    # 测试向量哈希
    v = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    hash_val = hash_vector(v)
    print(f"\nVector hash: {hash_val[:32]}...")

    # 测试内积保持验证
    p = np.random.randn(10)
    q = np.random.randn(10)
    p_enc = p + np.random.randn(10) * 0.1  # 模拟加密
    q_enc = q + np.random.randn(10) * 0.1

    preserved = verify_scalar_product_preservation(p_enc, q_enc, p, q, tolerance=0.5)
    print(f"\nScalar product preserved: {preserved}")

    # 测试统计分析
    original = np.random.randn(100, 10)
    encrypted = original + np.random.randn(100, 10) * 0.5

    stats = statistical_test_encryption(encrypted, original)
    print(f"\nStatistical test results:")
    print(f"  Encrypted mean: {stats['encrypted_mean']:.4f}")
    print(f"  Encrypted std: {stats['encrypted_std']:.4f}")
    print(f"  Correlation: {stats['correlation']:.4f}")
    print(f"  Encrypted entropy: {stats['encrypted_entropy']:.4f}")

    # 测试存储估计
    storage = estimate_storage_size(10000, 4096)
    print(f"\nEstimated storage for 10K vectors: {storage}")
