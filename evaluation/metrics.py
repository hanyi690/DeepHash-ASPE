
"""
检索评估指标

实现标准的检索评估指标：
- 平均精度均值 (mAP)
- Precision@K
- Recall@K
- NDCG@K
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from sklearn.metrics import ndcg_score


def compute_average_precision(retrieved_labels: np.ndarray,
                               relevant_labels: List) -> float:
    """
    计算单个查询的平均精度。

    参数：
        retrieved_labels: [N] 按顺序检索到的项的标签
        relevant_labels: 相关标签列表

    返回：
        平均精度分数
    """
    # 将相关标签转换为集合以便快速查找
    relevant_set = set(relevant_labels)

    # 跟踪每个相关项位置的精度
    precisions = []
    num_relevant = 0

    for i, label in enumerate(retrieved_labels):
        if label in relevant_set:
            num_relevant += 1
            precision = num_relevant / (i + 1)
            precisions.append(precision)

    # AP 是相关位置处精度的均值
    if len(precisions) == 0:
        return 0.0

    return np.mean(precisions)


def compute_map(query_labels: np.ndarray,
                retrieved_indices: np.ndarray,
                db_labels: np.ndarray,
                k: Optional[int] = None) -> float:
    """
    计算检索的平均精度均值。

    参数：
        query_labels: [N] 查询标签
        retrieved_indices: [N, k] 每个查询的检索索引
        db_labels: [M] 数据库标签
        k: 可选的要考虑的 top-k（默认：全部）

    返回：
        平均精度均值
    """
    num_queries = len(query_labels)
    if k is None:
        k = retrieved_indices.shape[1]

    average_precisions = []

    for i in range(num_queries):
        query_label = query_labels[i]
        retrieved = retrieved_indices[i, :k]
        retrieved_labels = db_labels[retrieved]

        # 获取所有相关项（与查询标签相同）
        relevant_labels = [query_label]

        # 计算 AP
        ap = compute_average_precision(retrieved_labels, relevant_labels)
        average_precisions.append(ap)

    return np.mean(average_precisions)


def compute_precision_at_k(query_labels: np.ndarray,
                          retrieved_indices: np.ndarray,
                          db_labels: np.ndarray,
                          k: int) -> float:
    """
    计算检索的 Precision@K。

    参数：
        query_labels: [N] 查询标签
        retrieved_indices: [N, M] 每个查询的检索索引 (M >= k)
        db_labels: [M] 数据库标签
        k: 要考虑的 top-k

    返回：
        Precision@K 分数
    """
    num_queries = len(query_labels)
    precisions = []

    for i in range(num_queries):
        query_label = query_labels[i]
        retrieved = retrieved_indices[i, :k]
        retrieved_labels = db_labels[retrieved]

        # 计算相关项数量
        num_relevant = np.sum(retrieved_labels == query_label)
        precision = num_relevant / k
        precisions.append(precision)

    return np.mean(precisions)


def compute_recall_at_k(query_labels: np.ndarray,
                       retrieved_indices: np.ndarray,
                       db_labels: np.ndarray,
                       k: int) -> float:
    """
    计算检索的 Recall@K。

    参数：
        query_labels: [N] 查询标签
        retrieved_indices: [N, M] 每个查询的检索索引 (M >= k)
        db_labels: [M] 数据库标签
        k: 要考虑的 top-k

    返回：
        Recall@K 分数
    """
    num_queries = len(query_labels)
    recalls = []

    for i in range(num_queries):
        query_label = query_labels[i]
        retrieved = retrieved_indices[i, :k]
        retrieved_labels = db_labels[retrieved]

        # 计算检索中的相关项数量
        num_relevant = np.sum(retrieved_labels == query_label)

        # 计算数据库中相关项总数
        total_relevant = np.sum(db_labels == query_label)

        if total_relevant == 0:
            recall = 0.0
        else:
            recall = num_relevant / total_relevant

        recalls.append(recall)

    return np.mean(recalls)


def compute_ndcg_at_k(query_labels: np.ndarray,
                     retrieved_scores: np.ndarray,
                     db_labels: np.ndarray,
                     k: int) -> float:
    """
    计算 NDCG@K（归一化折扣累积增益）。

    参数：
        query_labels: [N] 查询标签
        retrieved_scores: [N, M] 每个查询的相似度分数
        db_labels: [M] 数据库标签
        k: 要考虑的 top-k

    返回：
        NDCG@K 分数
    """
    num_queries = len(query_labels)
    ndcg_scores = []

    for i in range(num_queries):
        query_label = query_labels[i]
        scores = retrieved_scores[i, :k]

        # 相关性：标签匹配则为 1，否则为 0
        top_k_indices = np.argsort(-scores)[:k]
        relevance = (db_labels[top_k_indices] == query_label).astype(int)

        # 理想排序（所有相关项在前）
        ideal_relevance = np.sort(relevance)[::-1]

        # DCG
        dcg = np.sum(relevance / np.log2(np.arange(2, k + 2)))

        # IDCG
        idcg = np.sum(ideal_relevance / np.log2(np.arange(2, k + 2)))

        if idcg == 0:
            ndcg = 0.0
        else:
            ndcg = dcg / idcg

        ndcg_scores.append(ndcg)

    return np.mean(ndcg_scores)


def compute_retrieval_metrics(query_features: np.ndarray,
                             db_features: np.ndarray,
                             query_labels: np.ndarray,
                             db_labels: np.ndarray,
                             k_values: List[int] = [1, 5, 10, 20, 50],
                             metric: str = 'cosine') -> Dict[str, float]:
    """
    计算综合检索指标。

    参数：
        query_features: [N, d] 查询特征向量
        db_features: [M, d] 数据库特征向量
        query_labels: [N] 查询标签
        db_labels: [M] 数据库标签
        k_values: 用于精度/召回率的 k 值
        metric: 相似度度量（'cosine' 或 'inner'）

    返回：
        指标字典
    """
    # 计算相似度矩阵
    if metric == 'cosine':
        # 归一化特征
        query_features = query_features / (np.linalg.norm(query_features, axis=1, keepdims=True) + 1e-8)
        db_features = db_features / (np.linalg.norm(db_features, axis=1, keepdims=True) + 1e-8)

    similarities = np.dot(query_features, db_features.T)

    # 按相似度排序（降序）
    retrieved_indices = np.argsort(-similarities, axis=1)

    metrics = {}

    # mAP
    metrics['mAP'] = compute_map(query_labels, retrieved_indices, db_labels)

    # 每个k的 Precision@K 和 Recall@K
    for k in k_values:
        if k <= similarities.shape[1]:
            metrics[f'precision@{k}'] = compute_precision_at_k(
                query_labels, retrieved_indices, db_labels, k
            )
            metrics[f'recall@{k}'] = compute_recall_at_k(
                query_labels, retrieved_indices, db_labels, k
            )

    # NDCG@K
    for k in [10, 20]:
        if k <= similarities.shape[1]:
            metrics[f'ndcg@{k}'] = compute_ndcg_at_k(
                query_labels, similarities, db_labels, k
            )

    return metrics


def evaluate_cross_modal_retrieval(image_features: np.ndarray,
                                  text_features: np.ndarray,
                                  image_labels: np.ndarray,
                                  text_labels: np.ndarray,
                                  k_values: List[int] = [1, 5, 10]) -> Dict[str, Dict]:
    """
    评估跨模态检索（图像↔文本）。

    参数：
        image_features: [N_img, d] 图像特征
        text_features: [N_txt, d] 文本特征
        image_labels: [N_img] 图像标签
        text_labels: [N_txt] 文本标签
        k_values: 用于评估的 k 值

    返回：
        包含每个方向指标的字典
    """
    results = {}

    # 图像 -> 文本检索
    print("正在评估图像 -> 文本检索...")
    i2t_metrics = compute_retrieval_metrics(
        image_features, text_features,
        image_labels, text_labels,
        k_values=k_values
    )
    results['image_to_text'] = i2t_metrics

    # 文本 -> 图像检索
    print("正在评估文本 -> 图像检索...")
    t2i_metrics = compute_retrieval_metrics(
        text_features, image_features,
        text_labels, image_labels,
        k_values=k_values
    )
    results['text_to_image'] = t2i_metrics

    return results


def compute_hash_quality(hash_codes: np.ndarray) -> Dict[str, float]:
    """
    计算二进制哈希码的质量指标。

    参数：
        hash_codes: [N, bits] 二进制哈希码

    返回：
        质量指标字典
    """
    metrics = {}

    # 平衡性：衡量每一位的平衡程度
    # 理想平衡：每一位为 0.5 的 1 和 0.5 的 0
    bit_means = np.mean(hash_codes, axis=0)  # 对于平衡的 -1/+1 码应该约为 0
    balance = 1.0 - np.mean(np.abs(bit_means))
    metrics['balance'] = balance

    # 稀疏性：零位的比例（对于非二进制码）
    # 对于二进制码，这衡量零的数量
    sparsity = np.mean(hash_codes == 0)
    metrics['sparsity'] = sparsity

    # 唯一性：唯一哈希码的比例
    unique_codes = len(set(tuple(code) for code in hash_codes))
    uniqueness = unique_codes / len(hash_codes)
    metrics['uniqueness'] = uniqueness

    # 随机对之间的平均汉明距离
    sample_size = min(1000, len(hash_codes))
    sample_indices = np.random.choice(len(hash_codes), sample_size, replace=False)
    sample_codes = hash_codes[sample_indices]

    hamming_distances = []
    for i in range(sample_size):
        for j in range(i + 1, sample_size):
            # 汉明距离
            dist = np.sum(sample_codes[i] != sample_codes[j])
            hamming_distances.append(dist / hash_codes.shape[1])

    metrics['avg_hamming_distance'] = np.mean(hamming_distances)

    return metrics


if __name__ == "__main__":
    # 测试评估指标
    print("正在测试评估指标")

    # 生成虚拟数据
    num_queries = 100
    num_db = 500
    feature_dim = 128

    query_features = np.random.randn(num_queries, feature_dim)
    db_features = np.random.randn(num_db, feature_dim)

    # 创建标签（10个类别）
    num_classes = 10
    query_labels = np.random.randint(0, num_classes, num_queries)
    db_labels = np.random.randint(0, num_classes, num_db)

    # 测试检索指标
    print("\n=== 检索指标 ===")
    metrics = compute_retrieval_metrics(
        query_features, db_features,
        query_labels, db_labels,
        k_values=[1, 5, 10]
    )

    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    # 测试跨模态评估
    print("\n=== 跨模态评估 ===")
    image_features = np.random.randn(50, feature_dim)
    text_features = np.random.randn(100, feature_dim)
    image_labels = np.random.randint(0, 5, 50)
    text_labels = np.random.randint(0, 5, 100)

    cross_modal_results = evaluate_cross_modal_retrieval(
        image_features, text_features,
        image_labels, text_labels,
        k_values=[1, 5, 10]
    )

    for direction, metrics in cross_modal_results.items():
        print(f"\n{direction}:")
        for name, value in metrics.items():
            print(f"  {name}: {value:.4f}")

    # 测试哈希质量
    print("\n=== 哈希质量指标 ===")
    hash_codes = np.random.choice([-1, 1], size=(100, 64))
    quality = compute_hash_quality(hash_codes)

    for name, value in quality.items():
        print(f"{name}: {value:.4f}")

    print("\n评估指标测试完成！")
