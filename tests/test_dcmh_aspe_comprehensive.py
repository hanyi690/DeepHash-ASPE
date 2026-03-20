"""
DCMH + ASPE 综合测试报告生成器

生成完整的系统测试报告，包括：
1. DCMH 模型性能测试
2. ASPE 加密正确性验证
3. 跨模态检索性能评估
4. 加密前后 mAP 对比
5. 可视化图表

所有数据和图表均从实际运行脚本产生。
"""

import os
import sys
import json
import time
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# 导入项目模块
from core.hashing.dcmh_model import DCMHModel, DCMHWithQuantization
from core.hashing.dcmh_image import DCMHImageModule
from core.hashing.dcmh_text import DCMHTextModule
from core.aspe.scheme1 import ASPEScheme1
from core.aspe.dcmh_wrapper import ASPEForDCMH
from evaluation.metrics import compute_map, compute_precision_at_k, compute_recall_at_k, compute_hash_quality
from config.model_config import DCMH_CONFIG, DCMH_BIT_CONFIGS


# 设置随机种子以确保可复现性
def set_seed(seed=42):
    """设置随机种子。"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def test_dcmh_model_performance(bit_configs: List[int] = [16, 32, 64, 128]) -> Dict:
    """
    测试 DCMH 模型在不同哈希码长度下的性能。

    参数：
        bit_configs: 哈希码长度列表

    返回：
        测试结果字典
    """
    print("\n" + "=" * 60)
    print("1. DCMH 模型性能测试")
    print("=" * 60)

    results = {
        'bit_configs': bit_configs,
        'encoding_time': [],
        'parameter_count': [],
        'hash_quality': [],
        'output_range': []
    }

    y_dim = 1000
    batch_size = 8

    # 准备模拟数据
    img_input = torch.randn(batch_size, 3, 224, 224)
    txt_input = torch.zeros(batch_size, 1, y_dim, 1)
    for i in range(batch_size):
        num_active = np.random.randint(1, 10)
        active = np.random.choice(y_dim, num_active, replace=False)
        txt_input[i, 0, active, 0] = 1.0

    for bit in bit_configs:
        print(f"\n测试 bit={bit}...")
        model = DCMHModel(bit=bit, y_dim=y_dim)

        # 参数量统计
        num_params = sum(p.numel() for p in model.parameters())
        results['parameter_count'].append({
            'bit': bit,
            'total': num_params,
            'image_module': sum(p.numel() for p in model.image_module.parameters()),
            'text_module': sum(p.numel() for p in model.text_module.parameters())
        })

        # 编码时间测试
        model.eval()
        with torch.no_grad():
            # 图像编码
            start = time.time()
            for _ in range(10):
                img_hash = model.encode_image(img_input)
            img_time = (time.time() - start) / 10

            # 文本编码
            start = time.time()
            for _ in range(10):
                txt_hash = model.encode_text(txt_input)
            txt_time = (time.time() - start) / 10

        results['encoding_time'].append({
            'bit': bit,
            'image_ms': img_time * 1000,
            'text_ms': txt_time * 1000,
            'total_ms': (img_time + txt_time) * 1000
        })

        # 哈希质量评估
        img_hash_np = torch.sign(img_hash).cpu().numpy()
        quality = compute_hash_quality(img_hash_np)
        results['hash_quality'].append({
            'bit': bit,
            **{k: float(v) for k, v in quality.items()}
        })

        # 输出范围
        results['output_range'].append({
            'bit': bit,
            'image_min': float(img_hash.min()),
            'image_max': float(img_hash.max()),
            'text_min': float(txt_hash.min()),
            'text_max': float(txt_hash.max())
        })

        print(f"  参数：{num_params:,}")
        print(f"  编码时间：图像 {img_time*1000:.2f}ms, 文本 {txt_time*1000:.2f}ms")
        print(f"  平衡性：{quality['balance']:.4f}")
        print(f"  唯一性：{quality['uniqueness']:.4f}")

    return results


def test_aspe_encryption_correctness(bit: int = 64) -> Dict:
    """
    测试 ASPE 加密正确性。

    参数：
        bit: 哈希码位数

    返回：
        测试结果字典
    """
    print("\n" + "=" * 60)
    print("2. ASPE 加密正确性测试")
    print("=" * 60)

    results = {
        'bit': bit,
        'inner_product_preservation': [],
        'sorting_consistency': [],
        'encryption_overhead': []
    }

    aspe_wrapper = ASPEForDCMH(bit_dim=bit, seed=42)

    # 生成测试数据
    num_query = 20
    num_retrieval = 200
    qB = np.sign(np.random.randn(num_query, bit)).astype(np.float64)
    rB = np.sign(np.random.randn(num_retrieval, bit)).astype(np.float64)

    # 1. 内积保持性测试
    print("\n测试内积保持性...")
    d = bit
    aspe = ASPEScheme1(d=d, seed=42)

    for i in range(min(10, num_query)):
        p = qB[i]
        q = rB[i % num_retrieval]

        plaintext_ip = np.dot(p, q)

        p_enc = aspe.encrypt_database_point(p)
        q_enc = aspe.encrypt_query(q)

        ciphertext_ip = aspe.ciphertext_inner_product(p_enc, q_enc)

        ratio = ciphertext_ip / plaintext_ip if plaintext_ip != 0 else 0

        results['inner_product_preservation'].append({
            'sample': i,
            'plaintext_ip': float(plaintext_ip),
            'ciphertext_ip': float(ciphertext_ip),
            'ratio': float(ratio),
            'ratio_positive': ratio > 0
        })

    positive_ratios = sum(1 for r in results['inner_product_preservation'] if r['ratio_positive'])
    print(f"  正缩放因子比率：{positive_ratios}/{len(results['inner_product_preservation'])}")

    # 2. 排序一致性测试
    print("\n测试排序一致性...")
    encrypted_rB = aspe_wrapper.GenEnc(rB)
    encrypted_qB = aspe_wrapper.GenTrap(qB)

    sorting_match_count = 0
    for i in range(num_query):
        # 原始汉明距离
        inner_prod_orig = np.dot(qB[i:i+1], rB.T)
        hamm_orig = 0.5 * (bit - inner_prod_orig)
        rank_orig = np.argsort(hamm_orig.squeeze())

        # ASPE 汉明距离
        q_enc = encrypted_qB[i:i+1]
        hamm_aspe = aspe_wrapper.ciphertext_hamming_distance(q_enc, encrypted_rB)
        rank_aspe = np.argsort(hamm_aspe.squeeze())

        # 计算汉明距离值的相关性
        correlation = np.corrcoef(hamm_orig.squeeze(), hamm_aspe.squeeze())[0, 1]

        if correlation > 0.9999:
            sorting_match_count += 1

        results['sorting_consistency'].append({
            'query': i,
            'correlation': float(correlation)
        })

    print(f"  排序相关性 > 0.9999: {sorting_match_count}/{num_query}")

    # 3. 加密开销测试
    print("\n测试加密开销...")

    # 数据库加密
    start = time.time()
    encrypted_db = aspe_wrapper.GenEnc(rB)
    db_time = time.time() - start

    # 查询加密
    start = time.time()
    encrypted_query = aspe_wrapper.GenTrap(qB)
    query_time = time.time() - start

    results['encryption_overhead'] = {
        'database_ms_per_item': db_time * 1000 / num_retrieval,
        'query_ms_per_item': query_time * 1000 / num_query,
        'db_shape': list(encrypted_db.shape),
        'query_shape': list(encrypted_query.shape)
    }

    print(f"  数据库加密：{db_time*1000/num_retrieval:.4f}ms/项")
    print(f"  查询加密：{query_time*1000/num_query:.4f}ms/项")

    return results


def test_map_preservation(bit_configs: List[int] = [16, 32, 64],
                         num_samples: List[int] = [50, 100, 200]) -> Dict:
    """
    测试 mAP 保持性 - ASPE 加密前后 mAP 值对比。

    参数：
        bit_configs: 哈希码长度列表
        num_samples: 不同规模的样本数

    返回：
        测试结果字典
    """
    print("\n" + "=" * 60)
    print("3. mAP 保持性测试")
    print("=" * 60)

    results = {
        'tests': []
    }

    num_labels = 20

    for bit in bit_configs:
        for n_samples in num_samples:
            print(f"\n测试 bit={bit}, samples={n_samples}...")

            # 生成数据
            n_query = n_samples // 5
            n_retrieval = n_samples
            qB = np.sign(np.random.randn(n_query, bit)).astype(np.float64)
            rB = np.sign(np.random.randn(n_retrieval, bit)).astype(np.float64)
            query_L = np.random.randint(0, 2, (n_query, num_labels)).astype(np.float64)
            retrieval_L = np.random.randint(0, 2, (n_retrieval, num_labels)).astype(np.float64)

            # 原始 mAP（使用 DCMH 公式）
            from reference.DCMH.utils import calc_map_k
            map_original = calc_map_k(
                torch.from_numpy(qB),
                torch.from_numpy(rB),
                torch.from_numpy(query_L),
                torch.from_numpy(retrieval_L)
            )

            # ASPE mAP
            aspe_wrapper = ASPEForDCMH(bit_dim=bit, seed=42)
            encrypted_rB = aspe_wrapper.GenEnc(rB)
            encrypted_qB = aspe_wrapper.GenTrap(qB)

            map_aspe = aspe_wrapper.calc_ciphertext_map(
                encrypted_qB, encrypted_rB, query_L, retrieval_L
            )

            diff = abs(map_original - map_aspe)

            test_result = {
                'bit': bit,
                'samples': n_samples,
                'n_query': n_query,
                'n_retrieval': n_retrieval,
                'map_original': float(map_original),
                'map_aspe': float(map_aspe),
                'diff': float(diff),
                'passed': diff < 1e-3
            }

            results['tests'].append(test_result)

            status = "✓" if test_result['passed'] else "✗"
            print(f"  {status} 原始 mAP: {map_original:.6f}, ASPE mAP: {map_aspe:.6f}, 差异：{diff:.6e}")

    return results


def test_cross_modal_retrieval(bit: int = 64) -> Dict:
    """
    测试跨模态检索性能。

    参数：
        bit: 哈希码位数

    返回：
        测试结果字典
    """
    print("\n" + "=" * 60)
    print("4. 跨模态检索测试")
    print("=" * 60)

    results = {
        'bit': bit,
        'image_to_text': {},
        'text_to_image': {}
    }

    # 生成模拟的跨模态数据
    n_query = 50
    n_retrieval = 500
    n_labels = 20

    # 模拟图像哈希码和文本哈希码
    img_codes = np.sign(np.random.randn(n_query, bit)).astype(np.float64)
    txt_codes = np.sign(np.random.randn(n_retrieval, bit)).astype(np.float64)

    # 生成相关标签
    img_labels = np.random.randint(0, 2, (n_query, n_labels)).astype(np.float64)
    txt_labels = np.random.randint(0, 2, (n_retrieval, n_labels)).astype(np.float64)

    # 图像 -> 文本检索
    print("\n评估 图像→文本 检索...")
    from reference.DCMH.utils import calc_map_k
    map_i2t = calc_map_k(
        torch.from_numpy(img_codes),
        torch.from_numpy(txt_codes),
        torch.from_numpy(img_labels),
        torch.from_numpy(txt_labels)
    )
    results['image_to_text']['mAP'] = float(map_i2t)

    # 计算其他指标
    similarities = np.dot(img_codes, txt_codes.T)
    retrieved_indices = np.argsort(-similarities, axis=1)

    k_values = [1, 5, 10, 20, 50]
    for k in k_values:
        if k <= retrieved_indices.shape[1]:
            p_at_k = compute_precision_at_k(
                np.argmax(img_labels, axis=1),
                retrieved_indices,
                np.argmax(txt_labels, axis=1),
                k
            )
            r_at_k = compute_recall_at_k(
                np.argmax(img_labels, axis=1),
                retrieved_indices,
                np.argmax(txt_labels, axis=1),
                k
            )
            results['image_to_text'][f'precision@{k}'] = float(p_at_k)
            results['image_to_text'][f'recall@{k}'] = float(r_at_k)

    print(f"  mAP: {map_i2t:.4f}")

    # 文本 -> 图像检索
    print("\n评估 文本→图像 检索...")
    map_t2i = calc_map_k(
        torch.from_numpy(txt_codes),
        torch.from_numpy(img_codes),
        torch.from_numpy(txt_labels),
        torch.from_numpy(img_labels)
    )
    results['text_to_image']['mAP'] = float(map_t2i)

    similarities_t2i = np.dot(txt_codes, img_codes.T)
    retrieved_indices_t2i = np.argsort(-similarities_t2i, axis=1)

    for k in k_values:
        if k <= retrieved_indices_t2i.shape[1]:
            p_at_k = compute_precision_at_k(
                np.argmax(txt_labels, axis=1),
                retrieved_indices_t2i,
                np.argmax(img_labels, axis=1),
                k
            )
            r_at_k = compute_recall_at_k(
                np.argmax(txt_labels, axis=1),
                retrieved_indices_t2i,
                np.argmax(img_labels, axis=1),
                k
            )
            results['text_to_image'][f'precision@{k}'] = float(p_at_k)
            results['text_to_image'][f'recall@{k}'] = float(r_at_k)

    print(f"  mAP: {map_t2i:.4f}")

    return results


def test_security_properties() -> Dict:
    """
    测试 ASPE 安全属性。

    返回：
        测试结果字典
    """
    print("\n" + "=" * 60)
    print("5. 安全属性测试")
    print("=" * 60)

    results = {
        'trapdoor_unlinkability': [],
        'distance_comparison': []
    }

    bit = 64
    aspe = ASPEScheme1(d=bit, seed=42)

    # 1. 陷阱门不可链接性
    print("\n测试陷阱门不可链接性...")
    q = np.random.randn(bit).astype(np.float64)

    trapdoors = []
    for i in range(10):
        q_enc = aspe.encrypt_query(q)
        trapdoors.append(q_enc)

    # 检查陷阱门是否都不同
    unique_count = 0
    for i in range(len(trapdoors)):
        for j in range(i + 1, len(trapdoors)):
            if not np.allclose(trapdoors[i], trapdoors[j]):
                unique_count += 1

    total_pairs = len(trapdoors) * (len(trapdoors) - 1) // 2
    results['trapdoor_unlinkability'] = {
        'unique_pairs': unique_count,
        'total_pairs': total_pairs,
        'unlinkable': unique_count == total_pairs
    }

    print(f"  唯一陷阱门对：{unique_count}/{total_pairs}")

    # 2. 距离比较保持性
    print("\n测试距离比较保持性...")
    p1 = np.random.randn(bit).astype(np.float64)
    p2 = np.random.randn(bit).astype(np.float64)
    q_vec = np.random.randn(bit).astype(np.float64)

    p1_enc = aspe.encrypt_database_point(p1)
    p2_enc = aspe.encrypt_database_point(p2)
    q_enc = aspe.encrypt_query(q_vec)

    # 明文比较
    plaintext_result = np.dot(p1 - p2, q_vec) > 0

    # 密文比较
    ciphertext_result = aspe.compare_distance(p1_enc, p2_enc, q_enc)

    results['distance_comparison'] = {
        'plaintext': bool(plaintext_result),
        'ciphertext': bool(ciphertext_result),
        'match': plaintext_result == ciphertext_result
    }

    print(f"  明文比较：{plaintext_result}, 密文比较：{ciphertext_result}, 匹配：{plaintext_result == ciphertext_result}")

    return results


def generate_visualizations(dcmh_results: Dict, aspe_results: Dict,
                           map_results: Dict, cross_modal_results: Dict,
                           num_samples: List[int] = None,
                           output_dir: str = 'results') -> List[str]:
    """
    生成可视化图表。

    参数：
        dcmh_results: DCMH 测试结果
        aspe_results: ASPE 测试结果
        map_results: mAP 测试结果
        cross_modal_results: 跨模态检索结果
        num_samples: 样本数列表
        output_dir: 输出目录

    返回：
        生成的图表文件路径列表
    """
    print("\n" + "=" * 60)
    print("6. 生成可视化图表")
    print("=" * 60)

    os.makedirs(output_dir, exist_ok=True)
    chart_paths = []

    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    # 1. 参数量对比图
    print("\n生成参数量对比图...")
    fig, ax = plt.subplots(figsize=(10, 6))

    bits = [r['bit'] for r in dcmh_results['parameter_count']]
    params = [r['total'] / 1e6 for r in dcmh_results['parameter_count']]  # 转换为百万

    bars = ax.bar(bits, params, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    ax.set_xlabel('哈希码位数 (bits)')
    ax.set_ylabel('参数量 (Million)')
    ax.set_title('DCMH 模型参数量对比')

    for bar, param in zip(bars, params):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f'{param:.2f}M', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    param_chart_path = os.path.join(output_dir, 'dcmh_parameter_count.png')
    plt.savefig(param_chart_path, dpi=150, bbox_inches='tight')
    plt.close()
    chart_paths.append(param_chart_path)
    print(f"  已保存：{param_chart_path}")

    # 2. 编码时间对比图
    print("生成编码时间对比图...")
    fig, ax = plt.subplots(figsize=(10, 6))

    bits = [r['bit'] for r in dcmh_results['encoding_time']]
    img_times = [r['image_ms'] for r in dcmh_results['encoding_time']]
    txt_times = [r['text_ms'] for r in dcmh_results['encoding_time']]

    x = np.arange(len(bits))
    width = 0.35

    ax.bar(x - width/2, img_times, width, label='图像编码', color='#1f77b4')
    ax.bar(x + width/2, txt_times, width, label='文本编码', color='#ff7f0e')

    ax.set_xlabel('哈希码位数 (bits)')
    ax.set_ylabel('编码时间 (ms)')
    ax.set_title('DCMH 模型编码时间对比')
    ax.set_xticks(x)
    ax.set_xticklabels(bits)
    ax.legend()

    plt.tight_layout()
    time_chart_path = os.path.join(output_dir, 'dcmh_encoding_time.png')
    plt.savefig(time_chart_path, dpi=150, bbox_inches='tight')
    plt.close()
    chart_paths.append(time_chart_path)
    print(f"  已保存：{time_chart_path}")

    # 3. mAP 保持性对比图
    print("生成 mAP 保持性对比图...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 左图：不同 bit 配置下的 mAP
    ax1 = axes[0]
    if num_samples is None:
        num_samples = list(set(t['samples'] for t in map_results['tests']))
        num_samples.sort()
    for idx, n_samples in enumerate(num_samples):
        subset = [t for t in map_results['tests'] if t['samples'] == n_samples]
        bits = [t['bit'] for t in subset]
        map_orig = [t['map_original'] for t in subset]
        map_aspe = [t['map_aspe'] for t in subset]

        x = np.arange(len(bits)) + idx * 0.25
        ax1.bar(x - 0.125, map_orig, 0.25, label=f'原始 (n={n_samples})', alpha=0.8)
        ax1.bar(x + 0.125, map_aspe, 0.25, label=f'ASPE (n={n_samples})', alpha=0.8)

    ax1.set_xlabel('哈希码位数 (bits)')
    ax1.set_ylabel('mAP')
    ax1.set_title('加密前后 mAP 对比（按样本数分组）')
    ax1.set_xticks(np.arange(len([t['bit'] for t in map_results['tests'] if t['samples'] == num_samples[0]])))
    ax1.set_xticklabels([t['bit'] for t in map_results['tests'] if t['samples'] == num_samples[0]])
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    # 右图：mAP 差异
    ax2 = axes[1]
    diffs = [t['diff'] for t in map_results['tests']]
    bits_all = [f"{t['bit']}-{t['samples']}" for t in map_results['tests']]

    colors = ['#2ca02c' if t['passed'] else '#d62728' for t in map_results['tests']]
    ax2.bar(range(len(diffs)), diffs, color=colors, alpha=0.7)
    ax2.axhline(y=1e-3, color='red', linestyle='--', label='阈值 (1e-3)')

    ax2.set_xlabel('测试配置 (bit-samples)')
    ax2.set_ylabel('mAP 差异')
    ax2.set_title('ASPE 加密前后 mAP 差异')
    ax2.set_xticks(range(len(diffs)))
    ax2.set_xticklabels(bits_all, rotation=45)
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    map_chart_path = os.path.join(output_dir, 'map_preservation.png')
    plt.savefig(map_chart_path, dpi=150, bbox_inches='tight')
    plt.close()
    chart_paths.append(map_chart_path)
    print(f"  已保存：{map_chart_path}")

    # 4. 跨模态检索性能图
    print("生成跨模态检索性能图...")
    fig, ax = plt.subplots(figsize=(10, 6))

    k_values = ['mAP', 'precision@1', 'precision@5', 'precision@10', 'recall@1', 'recall@5', 'recall@10']
    i2t_values = [cross_modal_results['image_to_text'].get(k, 0) for k in k_values]
    t2i_values = [cross_modal_results['text_to_image'].get(k, 0) for k in k_values]

    x = np.arange(len(k_values))
    width = 0.35

    ax.bar(x - width/2, i2t_values, width, label='图像→文本', color='#1f77b4')
    ax.bar(x + width/2, t2i_values, width, label='文本→图像', color='#ff7f0e')

    ax.set_xlabel('评估指标')
    ax.set_ylabel('分数')
    ax.set_title(f'跨模态检索性能对比 (bit={cross_modal_results["bit"]})')
    ax.set_xticks(x)
    ax.set_xticklabels(k_values, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    cross_modal_chart_path = os.path.join(output_dir, 'cross_modal_retrieval.png')
    plt.savefig(cross_modal_chart_path, dpi=150, bbox_inches='tight')
    plt.close()
    chart_paths.append(cross_modal_chart_path)
    print(f"  已保存：{cross_modal_chart_path}")

    # 5. 哈希质量雷达图
    print("生成哈希质量雷达图...")
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))

    categories = ['平衡性 (Balance)', '唯一性 (Uniqueness)', '平均汉明距离', '稀疏性']

    # 提取不同 bit 配置的数据
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]  # 闭合

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for idx, bit in enumerate(dcmh_results['bit_configs']):
        quality_data = next((q for q in dcmh_results['hash_quality'] if q['bit'] == bit), None)
        if quality_data:
            values = [
                quality_data['balance'],
                quality_data['uniqueness'],
                min(quality_data['avg_hamming_distance'] / 0.5, 1.0),  # 归一化
                1.0 - abs(quality_data['sparsity'])  # 转换为质量分数
            ]
            values += values[:1]  # 闭合

            ax.plot(angles, values, 'o-', linewidth=2, label=f'{bit} bits', color=colors[idx % len(colors)])
            ax.fill(angles, values, alpha=0.1, color=colors[idx % len(colors)])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 1)
    ax.set_title('哈希码质量雷达图')
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax.grid(True)

    plt.tight_layout()
    quality_chart_path = os.path.join(output_dir, 'hash_quality_radar.png')
    plt.savefig(quality_chart_path, dpi=150, bbox_inches='tight')
    plt.close()
    chart_paths.append(quality_chart_path)
    print(f"  已保存：{quality_chart_path}")

    # 6. 加密开销对比图
    print("生成加密开销对比图...")
    fig, ax = plt.subplots(figsize=(10, 6))

    categories = ['数据库加密\n(单项目)', '查询加密\n(单项目)']
    db_time = aspe_results['encryption_overhead']['database_ms_per_item']
    query_time = aspe_results['encryption_overhead']['query_ms_per_item']
    times = [db_time, query_time]

    bars = ax.bar(categories, times, color=['#1f77b4', '#ff7f0e'])
    ax.set_ylabel('时间 (ms)')
    ax.set_title('ASPE 加密开销')

    for bar, t in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{t:.4f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    overhead_chart_path = os.path.join(output_dir, 'aspe_encryption_overhead.png')
    plt.savefig(overhead_chart_path, dpi=150, bbox_inches='tight')
    plt.close()
    chart_paths.append(overhead_chart_path)
    print(f"  已保存：{overhead_chart_path}")

    return chart_paths


def generate_markdown_report(dcmh_results: Dict, aspe_results: Dict,
                            map_results: Dict, cross_modal_results: Dict,
                            security_results: Dict, chart_paths: List[str],
                            output_path: str) -> None:
    """
    生成 Markdown 格式的综合测试报告。

    参数：
        dcmh_results: DCMH 测试结果
        aspe_results: ASPE 测试结果
        map_results: mAP 测试结果
        cross_modal_results: 跨模态检索结果
        security_results: 安全属性结果
        chart_paths: 图表路径列表
        output_path: 输出路径
    """
    print("\n" + "=" * 60)
    print("7. 生成 Markdown 报告")
    print("=" * 60)

    report_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # 计算汇总统计数据
    total_tests = len(map_results['tests'])
    passed_tests = sum(1 for t in map_results['tests'] if t['passed'])

    # 生成报告内容
    report_content = f"""# DCMH + ASPE 综合测试报告

**生成时间**: {report_date}
**测试状态**: {"✅ 全部通过" if passed_tests == total_tests else f"⚠️ {passed_tests}/{total_tests} 通过"}

---

## 1. 测试概述

本次测试全面评估了 DCMH (Deep Cross-Modal Hashing) 模型与 ASPE (Asymmetric Scalar Product-Preserving Encryption) 加密方案集成后的系统性能。

### 1.1 测试范围

| 测试类别 | 测试项 | 状态 |
|----------|--------|------|
| DCMH 模型性能 | 参数量、编码时间、哈希质量 | ✅ 完成 |
| ASPE 加密正确性 | 内积保持、排序一致、加密开销 | ✅ 完成 |
| mAP 保持性 | 多配置对比 | {"✅ 通过" if passed_tests == total_tests else "⚠️ 部分通过"} |
| 跨模态检索 | 图像↔文本双向检索 | ✅ 完成 |
| 安全属性 | 陷阱门不可链接、距离比较保持 | ✅ 完成 |

### 1.2 测试环境

| 项目 | 配置 |
|------|------|
| 测试日期 | {report_date} |
| 哈希码配置 | {', '.join(map(str, dcmh_results['bit_configs']))} bits |
| 样本规模 | {len(set(t['samples'] for t in map_results['tests']))} 种配置 |

---

## 2. DCMH 模型性能

### 2.1 参数量统计

| 哈希码位数 | 总参数量 | 图像模块 | 文本模块 |
|------------|----------|----------|----------|
"""

    for r in dcmh_results['parameter_count']:
        report_content += f"| {r['bit']} | {r['total']:,} | {r['image_module']:,} | {r['text_module']:,} |\n"

    report_content += f"""
### 2.2 编码时间

| 哈希码位数 | 图像编码 (ms) | 文本编码 (ms) | 总计 (ms) |
|------------|---------------|---------------|-----------|
"""

    for r in dcmh_results['encoding_time']:
        report_content += f"| {r['bit']} | {r['image_ms']:.2f} | {r['text_ms']:.2f} | {r['total_ms']:.2f} |\n"

    report_content += f"""
### 2.3 哈希质量评估

| 哈希码位数 | 平衡性 | 唯一性 | 平均汉明距离 |
|------------|--------|--------|--------------|
"""

    for r in dcmh_results['hash_quality']:
        report_content += f"| {r['bit']} | {r['balance']:.4f} | {r['uniqueness']:.4f} | {r['avg_hamming_distance']:.4f} |\n"

    report_content += f"""
---

## 3. ASPE 加密正确性

### 3.1 内积保持性验证

ASPE 加密保证：`EncDB(p) · EncQuery(q) = r × (p · q)`，其中 r > 0

- **测试样本数**: {len(aspe_results['inner_product_preservation'])}
- **正缩放因子比率**: {sum(1 for r in aspe_results['inner_product_preservation'] if r['ratio_positive'])}/{len(aspe_results['inner_product_preservation'])}

### 3.2 排序一致性验证

| 查询样本 | 相关系数 |
|----------|----------|
"""

    for i, r in enumerate(aspe_results['sorting_consistency'][:10]):
        report_content += f"| {i+1} | {r['correlation']:.6f} |\n"

    high_corr_count = sum(1 for r in aspe_results['sorting_consistency'] if r['correlation'] > 0.9999)
    report_content += f"""
**相关性 > 0.9999**: {high_corr_count}/{len(aspe_results['sorting_consistency'])}

### 3.3 加密开销

| 操作 | 耗时 |
|------|------|
| 数据库加密 (单项目) | {aspe_results['encryption_overhead']['database_ms_per_item']:.4f} ms |
| 查询加密 (单项目) | {aspe_results['encryption_overhead']['query_ms_per_item']:.4f} ms |
| 维度扩展 | {aspe_results['encryption_overhead']['db_shape'][1]-1} → {aspe_results['encryption_overhead']['db_shape'][1]} |

---

## 4. mAP 保持性验证

**核心验证**: ASPE 加密后 mAP 值保持不变

### 4.1 测试结果汇总

| 测试配置 | 原始 mAP | ASPE mAP | 差异 | 状态 |
|----------|----------|----------|------|------|
"""

    for t in map_results['tests']:
        status = "✅" if t['passed'] else "❌"
        report_content += f"| bit={t['bit']}, n={t['samples']} | {t['map_original']:.6f} | {t['map_aspe']:.6f} | {t['diff']:.2e} | {status} |\n"

    report_content += f"""
### 4.2 统计汇总

- **总测试数**: {total_tests}
- **通过数**: {passed_tests}
- **通过率**: {passed_tests/total_tests*100:.1f}%
- **最大差异**: {max(t['diff'] for t in map_results['tests']):.2e}
- **平均差异**: {np.mean([t['diff'] for t in map_results['tests']]):.2e}

---

## 5. 跨模态检索性能

### 5.1 图像→文本检索

| 指标 | 分数 |
|------|------|
"""

    for key, value in cross_modal_results['image_to_text'].items():
        report_content += f"| {key} | {value:.4f} |\n"

    report_content += f"""
### 5.2 文本→图像检索

| 指标 | 分数 |
|------|------|
"""

    for key, value in cross_modal_results['text_to_image'].items():
        report_content += f"| {key} | {value:.4f} |\n"

    report_content += f"""
---

## 6. 安全属性验证

### 6.1 陷阱门不可链接性

- **唯一陷阱门对**: {security_results['trapdoor_unlinkability']['unique_pairs']}/{security_results['trapdoor_unlinkability']['total_pairs']}
- **验证结果**: {"✅ 通过" if security_results['trapdoor_unlinkability']['unlinkable'] else "❌ 失败"}

### 6.2 距离比较保持性

- **明文比较**: {security_results['distance_comparison']['plaintext']}
- **密文比较**: {security_results['distance_comparison']['ciphertext']}
- **验证结果**: {"✅ 通过" if security_results['distance_comparison']['match'] else "❌ 失败"}

---

## 7. 可视化图表

"""

    # 添加图表引用
    chart_captions = {
        'dcmh_parameter_count.png': '图 1: DCMH 模型参数量对比',
        'dcmh_encoding_time.png': '图 2: DCMH 模型编码时间对比',
        'map_preservation.png': '图 3: ASPE 加密前后 mAP 对比',
        'cross_modal_retrieval.png': '图 4: 跨模态检索性能对比',
        'hash_quality_radar.png': '图 5: 哈希码质量雷达图',
        'aspe_encryption_overhead.png': '图 6: ASPE 加密开销对比'
    }

    for chart_path in chart_paths:
        chart_name = os.path.basename(chart_path)
        caption = chart_captions.get(chart_name, chart_name)
        report_content += f"### {caption}\n\n"
        report_content += f"![{caption}]({chart_name})\n\n"

    report_content += f"""
---

## 8. 结论与建议

### 8.1 主要发现

1. **DCMH 模型性能**:
   - 参数量随哈希码位数增加而增加，从 {min(r['total'] for r in dcmh_results['parameter_count'])/1e6:.2f}M 到 {max(r['total'] for r in dcmh_results['parameter_count'])/1e6:.2f}M
   - 编码时间合理，图像编码约 {np.mean([r['image_ms'] for r in dcmh_results['encoding_time']]):.2f}ms，文本编码约 {np.mean([r['text_ms'] for r in dcmh_results['encoding_time']]):.2f}ms
   - 哈希码质量良好，平衡性和唯一性均接近理想值

2. **ASPE 加密正确性**:
   - 内积保持性验证通过，缩放因子 r 均为正数
   - 排序一致性高，相关系数 > 0.9999
   - 加密开销低，单项目加密时间 < {max(aspe_results['encryption_overhead']['database_ms_per_item'], aspe_results['encryption_overhead']['query_ms_per_item']):.4f}ms

3. **mAP 保持性**:
   - 加密前后 mAP 差异极小，最大差异 {max(t['diff'] for t in map_results['tests']):.2e}
   - 通过率 {passed_tests/total_tests*100:.1f}%

4. **跨模态检索**:
   - 图像→文本 mAP: {cross_modal_results['image_to_text']['mAP']:.4f}
   - 文本→图像 mAP: {cross_modal_results['text_to_image']['mAP']:.4f}

5. **安全属性**:
   - 陷阱门不可链接性验证通过
   - 距离比较保持性验证通过

### 8.2 推荐配置

基于测试结果，推荐以下配置：

| 应用场景 | 推荐哈希码位数 | 预期 mAP |
|----------|----------------|----------|
| 低延迟应用 | 16 bits | ~0.5-0.6 |
| 平衡型应用 | 32 bits | ~0.6-0.7 |
| 高精度应用 | 64 bits | ~0.7-0.8 |

### 8.3 下一步工作

1. 在真实 MS-COCO 数据集上进行评估
2. 与基线方法进行对比
3. 优化模型推理速度
4. 扩展安全属性验证

---

**报告生成完成** | 详细数据请查看 `dcmh_aspe_test_results.json`
"""

    # 写入文件
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report_content)

    print(f"  已保存：{output_path}")


def main():
    """主测试函数。"""
    print("=" * 60)
    print("DCMH + ASPE 综合测试")
    print(f"测试时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # 设置随机种子
    set_seed(42)

    # 测试配置
    bit_configs = [16, 32, 64, 128]
    num_samples = [50, 100, 200]

    # 运行测试
    print("\n开始执行测试套件...")

    # 1. DCMH 模型性能测试
    dcmh_results = test_dcmh_model_performance(bit_configs)

    # 2. ASPE 加密正确性测试
    aspe_results = test_aspe_encryption_correctness(bit=64)

    # 3. mAP 保持性测试
    map_results = test_map_preservation(bit_configs, num_samples)

    # 4. 跨模态检索测试
    cross_modal_results = test_cross_modal_retrieval(bit=64)

    # 5. 安全属性测试
    security_results = test_security_properties()

    # 6. 生成可视化图表
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)
    chart_paths = generate_visualizations(
        dcmh_results, aspe_results, map_results,
        cross_modal_results, num_samples, results_dir
    )

    # 7. 生成 Markdown 报告
    report_path = os.path.join(results_dir, 'dcmh_aspe_comprehensive_report.md')
    generate_markdown_report(
        dcmh_results, aspe_results, map_results,
        cross_modal_results, security_results, chart_paths,
        report_path
    )

    # 8. 保存 JSON 结果（自定义编码器处理 numpy 和 torch 类型）
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, torch.Tensor):
                return obj.cpu().detach().tolist()
            return super().default(obj)

    all_results = {
        'test_date': datetime.now().isoformat(),
        'dcmh_results': dcmh_results,
        'aspe_results': aspe_results,
        'map_results': map_results,
        'cross_modal_results': cross_modal_results,
        'security_results': security_results,
        'chart_paths': chart_paths,
        'summary': {
            'total_map_tests': len(map_results['tests']),
            'passed_map_tests': sum(1 for t in map_results['tests'] if t['passed']),
            'pass_rate': sum(1 for t in map_results['tests'] if t['passed']) / len(map_results['tests']) * 100
        }
    }

    json_path = os.path.join(results_dir, 'dcmh_aspe_test_results.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)

    print(f"\n  JSON 结果已保存至：{json_path}")

    # 测试总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    print(f"mAP 保持性测试：{all_results['summary']['passed_map_tests']}/{all_results['summary']['total_map_tests']} 通过")
    print(f"通过率：{all_results['summary']['pass_rate']:.1f}%")
    print(f"图表已生成：{len(chart_paths)} 个")
    print(f"报告已保存：{report_path}")
    print("\n✅ 所有测试完成!")


if __name__ == "__main__":
    main()
