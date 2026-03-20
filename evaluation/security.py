"""
安全评估

评估 ASPE 加密方案的安全属性。
"""

import numpy as np
from typing import Dict, List, Tuple
from core.aspe.scheme1 import ASPEScheme1
from core.aspe.scheme2 import ASPEScheme2


def evaluate_scheme1_security(encrypted_db: np.ndarray,
                             known_points: np.ndarray,
                             known_encrypted: np.ndarray,
                             d: int) -> Dict:
    """
    评估方案 1 对 2 级攻击的抵抗能力。

    2 级攻击：攻击者知道一些原始样本，但不知道它们的加密版本。

    参数：
        encrypted_db: 加密的数据库点
        known_points: 已知原始点（未加密）
        known_encrypted: 已知点的加密版本
        d: 特征维度

    返回：
        包含安全指标的字典
    """
    results = {}

    # 测试1：统计分析
    # 计算加密数据的均值和标准差
    results['encrypted_mean'] = np.mean(encrypted_db)
    results['encrypted_std'] = np.std(encrypted_db)
    results['encrypted_min'] = np.min(encrypted_db)
    results['encrypted_max'] = np.max(encrypted_db)

    # 测试2：相关性分析
    # 衡量原始数据和加密数据之间的相关性
    correlations = []
    for i in range(len(known_points)):
        corr = np.corrcoef(known_points[i].flatten(),
                          known_encrypted[i].flatten())[0, 1]
        correlations.append(corr)

    results['avg_correlation'] = np.mean(correlations)
    results['max_correlation'] = np.max(correlations)

    # 测试3：重建难度
    # 尝试从加密数据重建原始数据（应该失败）
    reconstruction_errors = []
    for i in range(len(known_points)):
        # 简单的重建尝试
        error = np.linalg.norm(known_points[i] - known_encrypted[i, :d])
        reconstruction_errors.append(error)

    results['avg_reconstruction_error'] = np.mean(reconstruction_errors)

    # 测试4：距离保持性
    # 检查欧几里得距离是否可恢复（不应该可恢复）
    n_test = min(10, len(known_points))
    distance_ratios = []

    for i in range(n_test):
        for j in range(i + 1, n_test):
            # 原始距离
            orig_dist = np.linalg.norm(known_points[i] - known_points[j])

            # 加密距离（不应该成比例）
            enc_dist = np.linalg.norm(known_encrypted[i] - known_encrypted[j])

            if orig_dist > 1e-6:
                distance_ratios.append(enc_dist / orig_dist)

    # 比率的方差表明距离未被保持
    results['distance_ratio_variance'] = np.var(distance_ratios) if distance_ratios else 0
    results['distance_preservation'] = results['distance_ratio_variance'] < 0.1

    return results


def evaluate_scheme2_security(encrypted_db_a: np.ndarray,
                             encrypted_db_b: np.ndarray,
                             known_points: np.ndarray,
                             known_encrypted_a: np.ndarray,
                             known_encrypted_b: np.ndarray,
                             d: int,
                             d_prime: int) -> Dict:
    """
    评估方案 2 对 3 级攻击的抵抗能力。

    3 级攻击：攻击者知道一些明文-密文对。

    参数：
        encrypted_db_a: 加密数据库（份额 A）
        encrypted_db_b: 加密数据库（份额 B）
        known_points: 已知原始点
        known_encrypted_a: 已知点的加密版本（份额 A）
        known_encrypted_b: 已知点的加密版本（份额 B）
        d: 原始特征维度
        d_prime: 扩展维度

    返回：
        包含安全指标的字典
    """
    results = {}

    # 测试1：分割有效性
    # 检查单个份额是否泄露信息
    share_a_errors = []
    share_b_errors = []

    for i in range(len(known_points)):
        # 尝试仅从份额 A 重建
        error_a = np.linalg.norm(known_points[i] - known_encrypted_a[i, :d])
        share_a_errors.append(error_a)

        # 尝试仅从份额 B 重建
        error_b = np.linalg.norm(known_points[i] - known_encrypted_b[i, :d])
        share_b_errors.append(error_b)

    results['share_a_reconstruction_error'] = np.mean(share_a_errors)
    results['share_b_reconstruction_error'] = np.mean(share_b_errors)

    # 测试2：组合份额分析
    # 即使有两个份额，直接重建也应该失败
    # （在不知道分割配置的情况下）
    combined_errors = []
    for i in range(len(known_points)):
        combined = known_encrypted_a[i] + known_encrypted_b[i]
        error = np.linalg.norm(known_points[i] - combined[:d])
        combined_errors.append(error)

    results['combined_reconstruction_error'] = np.mean(combined_errors)

    # 测试3：人工维度分析
    # 检查人工维度是否掩盖了真实结构
    real_dims = d + 1
    artificial_dims_start = real_dims

    # 计算人工维度与真实维度的方差
    real_variance = np.var(encrypted_db_a[:, :real_dims])
    artificial_variance = np.var(encrypted_db_a[:, artificial_dims_start:])

    results['real_dimension_variance'] = real_variance
    results['artificial_dimension_variance'] = artificial_variance
    results['variance_ratio'] = artificial_variance / (real_variance + 1e-8)

    # 测试4：与分割的相关性
    # 衡量分割模式是否可恢复
    # （不应该仅从加密数据中恢复）
    results['splitting_entropy'] = np.log2(d_prime)  # 最大熵用于安全

    # 测试5：已知明文攻击抵抗能力
    # 尝试从已知对恢复变换矩阵
    # （简化测试 - 实际攻击会更复杂）
    results['known_plaintext_resistance'] = test_known_plaintext_resistance(
        known_points, known_encrypted_a, known_encrypted_b, d, d_prime
    )

    return results


def test_known_plaintext_resistance(known_points: np.ndarray,
                                   known_encrypted_a: np.ndarray,
                                   known_encrypted_b: np.ndarray,
                                   d: int,
                                   d_prime: int) -> Dict:
    """
    测试对已知明文攻击的抵抗能力。

    参数：
        known_points: 已知明文向量
        known_encrypted_a: 加密向量（份额 A）
        known_encrypted_b: 加密向量（份额 B）
        d: 原始维度
        d_prime: 扩展维度

    返回：
        抵抗能力指标
    """
    results = {}

    # 尝试求解变换（简化）
    # 实际上，这应该由于分割而失败

    # 线性回归尝试
    n_known = len(known_points)

    if n_known >= d:
        # 尝试找到线性关系（应该由于分割而失败）
        try:
            # 尝试最小二乘拟合
            # P * M ≈ P_enc（应该由于分割而不起作用）
            M_est_a, _, _, _ = np.linalg.lstsq(known_points, known_encrypted_a[:, :d], rcond=None)
            reconstruction_error_a = np.linalg.norm(
                known_points @ M_est_a - known_encrypted_a[:, :d]
            )

            M_est_b, _, _, _ = np.linalg.lstsq(known_points, known_encrypted_b[:, :d], rcond=None)
            reconstruction_error_b = np.linalg.norm(
                known_points @ M_est_b - known_encrypted_b[:, :d]
            )

            results['matrix_a_estimation_error'] = reconstruction_error_a
            results['matrix_b_estimation_error'] = reconstruction_error_b

            # 高误差表明良好的抵抗能力
            results['resistance_score'] = (reconstruction_error_a + reconstruction_error_b) / 2

        except Exception as e:
            results['estimation_failed'] = True
            results['resistance_score'] = float('inf')

    return results


def differential_privacy_analysis(encrypted_db: np.ndarray,
                                  original_db: np.ndarray,
                                  epsilon: float = 1.0) -> Dict:
    """
    分析差分隐私属性。

    参数：
        encrypted_db: 加密数据库
        original_db: 原始数据库
        epsilon: 隐私参数

    返回：
        差分隐私指标
    """
    results = {}

    # 计算敏感性
    sensitivities = []
    n_samples = min(100, len(original_db))

    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            # L2 敏感性
            sens = np.linalg.norm(encrypted_db[i] - encrypted_db[j])
            sensitivities.append(sens)

    results['avg_sensitivity'] = np.mean(sensitivities)
    results['max_sensitivity'] = np.max(sensitivities)

    # 估计隐私损失（简化）
    # 实际上，这需要正式分析
    results['estimated_epsilon'] = epsilon

    return results


def statistical_distance_analysis(original_db: np.ndarray,
                                  encrypted_db: np.ndarray) -> Dict:
    """
    计算原始分布和加密分布之间的统计距离。

    参数：
        original_db: 原始特征数据库
        encrypted_db: 加密数据库

    返回：
        统计距离指标
    """
    results = {}

    # 展平以进行分布分析
    orig_flat = original_db.flatten()
    enc_flat = encrypted_db.flatten()

    # KL 散度估计（简化）
    # 使用基于直方图的估计
    n_bins = 50

    orig_hist, orig_bins = np.histogram(orig_flat, bins=n_bins, density=True)
    enc_hist, enc_bins = np.histogram(enc_flat, bins=n_bins, density=True)

    # 避免零
    orig_hist = orig_hist + 1e-10
    enc_hist = enc_hist + 1e-10

    # KL 散度
    kl_div = np.sum(orig_hist * np.log(orig_hist / enc_hist))
    results['kl_divergence'] = kl_div

    # Jensen-Shannon 散度（对称）
    js_div = 0.5 * np.sum(orig_hist * np.log(orig_hist / (0.5 * orig_hist + 0.5 * enc_hist))) + \
             0.5 * np.sum(enc_hist * np.log(enc_hist / (0.5 * orig_hist + 0.5 * enc_hist)))
    results['js_divergence'] = js_div

    # 总变差距离
    tv_dist = 0.5 * np.sum(np.abs(orig_hist - enc_hist))
    results['total_variation_distance'] = tv_dist

    return results


if __name__ == "__main__":
    # 测试安全评估
    print("正在测试安全评估")

    # 生成测试数据
    d = 64
    n_samples = 100

    original_db = np.random.randn(n_samples, d)

    # 测试方案 1 的安全性
    print("\n=== 方案 1 安全评估 ===")
    scheme1 = ASPEScheme1(d=d)
    encrypted_db_scheme1 = scheme1.encrypt_database(original_db)

    known_points = original_db[:10]
    known_encrypted = encrypted_db_scheme1[:10]

    security_results = evaluate_scheme1_security(
        encrypted_db_scheme1, known_points, known_encrypted, d
    )

    for key, value in security_results.items():
        if isinstance(value, float):
            print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: {value}")

    # 测试方案 2 的安全性
    print("\n=== 方案 2 安全评估 ===")
    scheme2 = ASPEScheme2(d=d)
    enc_a, enc_b = scheme2.encrypt_database(original_db)

    known_enc_a = enc_a[:10]
    known_enc_b = enc_b[:10]

    security_results2 = evaluate_scheme2_security(
        enc_a, enc_b, known_points, known_enc_a, known_enc_b, d, scheme2.d_prime
    )

    for key, value in security_results2.items():
        if isinstance(value, float):
            print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: {value}")

    # 测试统计距离
    print("\n=== 统计距离分析 ===")
    distance_results = statistical_distance_analysis(
        original_db[:20], encrypted_db_scheme1[:20, :d]
    )

    for key, value in distance_results.items():
        print(f"{key}: {value:.6f}")

    print("\n安全评估测试完成！")
