"""
性能基准测试

对 ASPE 加密和检索操作进行性能基准测试。
"""

import time
import numpy as np
from typing import Dict, List
import matplotlib.pyplot as plt

from core.aspe.scheme1 import ASPEScheme1
from core.aspe.scheme2 import ASPEScheme2


def benchmark_encryption_time(points: np.ndarray,
                              scheme: str = 'scheme1',
                              **kwargs) -> Dict:
    """
    对数据库点的加密时间进行基准测试。

    参数：
        points: [N, d] 要加密的点数组
        scheme: 'scheme1' 或 'scheme2'
        **kwargs: ASPE 的附加参数

    返回：
        包含计时结果的字典
    """
    n, d = points.shape

    # 初始化 ASPE
    if scheme == 'scheme1':
        aspe = ASPEScheme1(d=d, **kwargs)
    else:
        aspe = ASPEScheme2(d=d, **kwargs)

    # 热身
    _ = aspe.encrypt_database_point(points[0])

    # 对单点加密进行基准测试
    times = []
    for _ in range(100):
        start = time.perf_counter()
        _ = aspe.encrypt_database_point(points[0])
        end = time.perf_counter()
        times.append(end - start)

    avg_single_time = np.mean(times)
    std_single_time = np.std(times)

    # 对批量加密进行基准测试
    start = time.perf_counter()
    if scheme == 'scheme1':
        _ = aspe.encrypt_database(points)
    else:
        _ = aspe.encrypt_database(points)
    end = time.perf_counter()

    batch_time = end - start
    throughput = n / batch_time

    return {
        'num_points': n,
        'dimension': d,
        'avg_single_time_ms': avg_single_time * 1000,
        'std_single_time_ms': std_single_time * 1000,
        'batch_time_s': batch_time,
        'throughput_points_per_sec': throughput,
        'scheme': scheme
    }


def benchmark_query_time(query: np.ndarray,
                        encrypted_db: np.ndarray,
                        scheme: str = 'scheme1',
                        **kwargs) -> Dict:
    """
    对查询时间进行基准测试（门限生成 + 内积）。

    参数：
        query: [d] 查询点
        encrypted_db: 加密数据库
        scheme: 'scheme1' 或 'scheme2'
        **kwargs: ASPE 的附加参数

    返回：
        包含计时结果的字典
    """
    d = query.shape[0]
    n_db = len(encrypted_db) if scheme == 'scheme1' else len(encrypted_db[0])

    # 初始化 ASPE（使用与数据库相同的种子）
    aspe_kwargs = kwargs.get('aspe_kwargs', {})
    if scheme == 'scheme1':
        aspe = ASPEScheme1(d=d, **aspe_kwargs)
    else:
        aspe = ASPEScheme2(d=d, **aspe_kwargs)

    # 对门限生成进行基准测试
    trapdoor_times = []
    for _ in range(100):
        start = time.perf_counter()
        trapdoor = aspe.encrypt_query(query)
        end = time.perf_counter()
        trapdoor_times.append(end - start)

    avg_trapdoor_time = np.mean(trapdoor_times)

    # 对内积计算进行基准测试
    if scheme == 'scheme1':
        inner_prod_times = []
        for _ in range(100):
            start = time.perf_counter()
            _ = aspe.ciphertext_inner_product(encrypted_db[0], trapdoor)
            end = time.perf_counter()
            inner_prod_times.append(end - start)

        avg_inner_time = np.mean(inner_prod_times)
    else:  # scheme2
        inner_prod_times = []
        for _ in range(100):
            start = time.perf_counter()
            _ = aspe.ciphertext_inner_product(
                (encrypted_db[0][0], encrypted_db[1][0]), trapdoor
            )
            end = time.perf_counter()
            inner_prod_times.append(end - start)

        avg_inner_time = np.mean(inner_prod_times)

    # 对完整查询进行基准测试（门限 + k 个内积）
    k = 100
    full_query_times = []

    for _ in range(100):
        start = time.perf_counter()
        trapdoor = aspe.encrypt_query(query)

        # 计算 top-k
        scores = []
        for i in range(k):
            if scheme == 'scheme1':
                score = aspe.ciphertext_inner_product(encrypted_db[i], trapdoor)
            else:
                score = aspe.ciphertext_inner_product(
                    (encrypted_db[0][i], encrypted_db[1][i]), trapdoor
                )
            scores.append(score)

        end = time.perf_counter()
        full_query_times.append(end - start)

    avg_full_time = np.mean(full_query_times)

    return {
        'dimension': d,
        'database_size': n_db,
        'avg_trapdoor_time_ms': avg_trapdoor_time * 1000,
        'avg_inner_product_time_ms': avg_inner_time * 1000,
        'avg_full_query_time_ms': avg_full_time * 1000,
        'top_k': k,
        'scheme': scheme
    }


def benchmark_scalability(d_values: List[int],
                         n_values: List[int],
                         scheme: str = 'scheme1') -> Dict:
    """
    对不同维度和数据库大小进行可扩展性基准测试。

    参数：
        d_values: 要测试的维度列表
        n_values: 要测试的数据库大小列表
        scheme: 'scheme1' 或 'scheme2'

    返回：
        包含可扩展性结果的字典
    """
    results = {
        'dimension_scalability': [],
        'size_scalability': []
    }

    # 测试维度可扩展性（固定数据库大小）
    fixed_n = 1000
    for d in d_values:
        print(f"正在测试维度 {d}...")

        points = np.random.randn(fixed_n, d)
        result = benchmark_encryption_time(points, scheme)
        results['dimension_scalability'].append(result)

    # 测试数据库大小可扩展性（固定维度）
    fixed_d = 512
    for n in n_values:
        print(f"正在测试数据库大小 {n}...")

        points = np.random.randn(n, fixed_d)
        result = benchmark_encryption_time(points, scheme)
        results['size_scalability'].append(result)

    return results


def compare_schemes(d: int = 512,
                   n: int = 1000) -> Dict:
    """
    比较方案 1 和方案 2 之间的性能。

    参数：
        d: 特征维度
        n: 数据库大小

    返回：
        比较结果
    """
    points = np.random.randn(n, d)

    # 方案 1 基准测试
    print("正在对方案 1 进行基准测试...")
    scheme1_results = benchmark_encryption_time(points, 'scheme1', seed=42)

    # 方案 2 基准测试
    print("正在对方案 2 进行基准测试...")
    scheme2_results = benchmark_encryption_time(points, 'scheme2', seed=42)

    # 查询基准测试
    query = np.random.randn(d)

    # 为查询基准测试初始化加密数据库
    scheme1 = ASPEScheme1(d=d, seed=42)
    enc_db1 = scheme1.encrypt_database(points)

    scheme2 = ASPEScheme2(d=d, seed=42)
    enc_db2 = scheme2.encrypt_database(points)

    scheme1_query = benchmark_query_time(
        query, enc_db1, 'scheme1', aspe_kwargs={'seed': 42}
    )

    scheme2_query = benchmark_query_time(
        query, enc_db2, 'scheme2', aspe_kwargs={'seed': 42}
    )

    return {
        'encryption': {
            'scheme1': scheme1_results,
            'scheme2': scheme2_results
        },
        'query': {
            'scheme1': scheme1_query,
            'scheme2': scheme2_query
        }
    }


def plot_benchmark_results(results: Dict, save_path: str = None):
    """
    绘制基准测试结果。

    参数：
        results: 基准测试结果
        save_path: 保存图表的可选路径
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 维度可扩展性
    ax = axes[0, 0]
    dim_results = results['dimension_scalability']
    dims = [r['dimension'] for r in dim_results]
    times = [r['batch_time_s'] for r in dim_results]

    ax.plot(dims, times, marker='o')
    ax.set_xlabel('维度')
    ax.set_ylabel('批量加密时间（秒）')
    ax.set_title('维度可扩展性')
    ax.grid(True)

    # 数据库大小可扩展性
    ax = axes[0, 1]
    size_results = results['size_scalability']
    sizes = [r['num_points'] for r in size_results]
    throughputs = [r['throughput_points_per_sec'] for r in size_results]

    ax.plot(sizes, throughputs, marker='o')
    ax.set_xlabel('数据库大小')
    ax.set_ylabel('吞吐量（点/秒）')
    ax.set_title('数据库大小可扩展性')
    ax.grid(True)

    # 方案比较 - 加密
    ax = axes[1, 0]
    schemes = ['方案 1', '方案 2']
    enc_times = [
        results['encryption']['scheme1']['avg_single_time_ms'],
        results['encryption']['scheme2']['avg_single_time_ms']
    ]

    ax.bar(schemes, enc_times)
    ax.set_ylabel('时间（毫秒）')
    ax.set_title('单点加密时间')
    ax.grid(True, axis='y')

    # 方案比较 - 查询
    ax = axes[1, 1]
    query_times = [
        results['query']['scheme1']['avg_full_query_time_ms'],
        results['query']['scheme2']['avg_full_query_time_ms']
    ]

    ax.bar(schemes, query_times)
    ax.set_ylabel('时间（毫秒）')
    ax.set_title('查询时间（门限 + top-100）')
    ax.grid(True, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"图表已保存到 {save_path}")

    plt.show()


if __name__ == "__main__":
    # 测试基准测试
    print("性能基准测试")

    # 测试单个基准
    print("\n=== 单个基准测试 ===")
    d = 256
    n = 100

    points = np.random.randn(n, d)

    result = benchmark_encryption_time(points, 'scheme1')
    for key, value in result.items():
        if isinstance(value, float):
            print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: {value}")

    # 比较方案
    print("\n=== 方案比较 ===")
    comparison = compare_schemes(d=256, n=500)

    print("\n加密性能：")
    for scheme in ['scheme1', 'scheme2']:
        print(f"\n{scheme.upper()}:")
        for key, value in comparison['encryption'][scheme].items():
            if isinstance(value, float):
                print(f"  {key}: {value:.6f}")

    print("\n查询性能：")
    for scheme in ['scheme1', 'scheme2']:
        print(f"\n{scheme.upper()}:")
        for key, value in comparison['query'][scheme].items():
            if isinstance(value, float):
                print(f"  {key}: {value:.6f}")

    # 可扩展性测试
    print("\n=== 可扩展性测试 ===")
    d_values = [64, 128, 256, 512, 1024]
    n_values = [100, 500, 1000, 5000, 10000]

    scalability = benchmark_scalability(
        d_values=[64, 128, 256],
        n_values=[100, 500, 1000],
        scheme='scheme1'
    )

    print("\n维度可扩展性：")
    for r in scalability['dimension_scalability']:
        print(f"  d={r['dimension']}: {r['throughput_points_per_sec']:.0f} 点/秒")

    print("\n大小可扩展性：")
    for r in scalability['size_scalability']:
        print(f"  n={r['num_points']}: {r['batch_time_s']:.3f}秒 用于加密")

    print("\n基准测试完成！")
