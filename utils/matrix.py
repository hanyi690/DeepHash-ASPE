"""
矩阵运算工具

用于ASPE加密的矩阵运算工具函数。
"""

import numpy as np
from typing import Tuple


def is_invertible(M: np.ndarray, tol: float = 1e-6) -> bool:
    """
    检查矩阵是否可逆。

    Args:
        M: 要检查的矩阵
        tol: 行列式的容差

    Returns:
        如果矩阵可逆则返回True
    """
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        return False

    try:
        det = np.linalg.det(M)
        return abs(det) > tol
    except np.linalg.LinAlgError:
        return False


def generate_orthogonal_matrix(n: int, seed: int = None) -> np.ndarray:
    """
    使用QR分解生成正交矩阵。

    Args:
        n: 矩阵维度
        seed: 随机种子

    Returns:
        n×n正交矩阵
    """
    if seed is not None:
        np.random.seed(seed)

    # 生成随机矩阵
    A = np.random.randn(n, n)

    # QR分解
    Q, R = np.linalg.qr(A)

    # 确保R的对角线为正
    D = np.diag(np.sign(np.diag(R)))
    Q = np.dot(Q, D)

    return Q


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    """
    L2归一化向量。

    Args:
        vectors: [N, d]向量数组

    Returns:
        [N, d]归一化向量数组
    """
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    # 避免除零
    norms[norms == 0] = 1.0
    return vectors / norms


def compute_pairwise_distances(matrix1: np.ndarray,
                               matrix2: np.ndarray = None,
                               metric: str = 'euclidean') -> np.ndarray:
    """
    计算向量之间的成对距离。

    Args:
        matrix1: [N, d]向量数组
        matrix2: [M, d]向量数组(如果为None,计算matrix1与自身的距离)
        metric: 距离度量('euclidean', 'cosine', 'inner')

    Returns:
        [N, M]距离矩阵
    """
    if matrix2 is None:
        matrix2 = matrix1

    if metric == 'euclidean':
        # 欧氏距离: ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a·b
        norm1 = np.sum(matrix1 ** 2, axis=1, keepdims=True)
        norm2 = np.sum(matrix2 ** 2, axis=1, keepdims=True)
        distances = norm1 + norm2.T - 2 * np.dot(matrix1, matrix2.T)
        # 确保非负(处理数值误差)
        distances = np.maximum(distances, 0)
        return np.sqrt(distances)

    elif metric == 'cosine':
        # 余弦距离: 1 - cosine_similarity
        similarities = np.dot(matrix1, matrix2.T)
        norm1 = np.linalg.norm(matrix1, axis=1, keepdims=True)
        norm2 = np.linalg.norm(matrix2, axis=1, keepdims=True)
        similarities = similarities / (norm1 * norm2.T + 1e-8)
        return 1 - similarities

    elif metric == 'inner':
        # 内积(负值用于距离转换)
        return -np.dot(matrix1, matrix2.T)

    else:
        raise ValueError(f"Unknown metric: {metric}")


def compute_pairwise_similarities(vectors1: np.ndarray,
                                  vectors2: np.ndarray = None,
                                  metric: str = 'inner') -> np.ndarray:
    """
    计算向量之间的成对相似度。

    Args:
        vectors1: [N, d]向量数组
        vectors2: [M, d]向量数组(如果为None,计算vectors1与自身的相似度)
        metric: 相似度度量('inner', 'cosine')

    Returns:
        [N, M]相似度矩阵
    """
    if vectors2 is None:
        vectors2 = vectors1

    if metric == 'inner':
        return np.dot(vectors1, vectors2.T)

    elif metric == 'cosine':
        # 余弦相似度
        similarities = np.dot(vectors1, vectors2.T)
        norm1 = np.linalg.norm(vectors1, axis=1, keepdims=True)
        norm2 = np.linalg.norm(vectors2, axis=1, keepdims=True)
        return similarities / (norm1 * norm2.T + 1e-8)

    else:
        raise ValueError(f"Unknown metric: {metric}")


def find_k_nearest_neighbors(query: np.ndarray,
                             database: np.ndarray,
                             k: int,
                             metric: str = 'inner') -> Tuple[np.ndarray, np.ndarray]:
    """
    在数据库中为每个查询查找k近邻。

    Args:
        query: [N, d]查询向量
        database: [M, d]数据库向量
        k: 邻居数量
        metric: 相似度度量('inner', 'cosine', 'euclidean')

    Returns:
        (indices, distances)元组:
        - indices: [N, k]邻居索引数组
        - distances: [N, k]距离/相似度数组
    """
    if metric == 'euclidean':
        distances = compute_pairwise_distances(query, database, 'euclidean')
        # 对于距离,我们想要最小值
        indices = np.argpartition(distances, k-1, axis=1)[:, :k]
        # 对k个值进行排序
        indices_sorted = np.zeros_like(indices)
        distances_sorted = np.zeros((indices.shape[0], k))
        for i in range(indices.shape[0]):
            idx = indices[i]
            dist = distances[i, idx]
            sorted_order = np.argsort(dist)
            indices_sorted[i] = idx[sorted_order]
            distances_sorted[i] = dist[sorted_order]
        return indices_sorted, distances_sorted

    else:  # inner or cosine (similarity metrics)
        similarities = compute_pairwise_similarities(query, database, metric)
        # 对于相似度,我们想要最大值
        indices = np.argpartition(-similarities, k-1, axis=1)[:, :k]
        # 对k个值进行排序
        indices_sorted = np.zeros_like(indices)
        similarities_sorted = np.zeros((indices.shape[0], k))
        for i in range(indices.shape[0]):
            idx = indices[i]
            sim = similarities[i, idx]
            sorted_order = np.argsort(-sim)
            indices_sorted[i] = idx[sorted_order]
            similarities_sorted[i] = sim[sorted_order]
        return indices_sorted, similarities_sorted


def rank_vectors(query: np.ndarray,
                 vectors: np.ndarray,
                 metric: str = 'inner') -> np.ndarray:
    """
    按与查询的相似度对向量排序。

    Args:
        query: [d]查询向量
        vectors: [N, d]向量数组
        metric: 相似度度量

    Returns:
        [N]按相似度排序的索引数组(降序)
    """
    if metric == 'inner':
        similarities = np.dot(vectors, query)
    elif metric == 'cosine':
        similarities = np.dot(vectors, query)
        similarities = similarities / (np.linalg.norm(vectors, axis=1) * np.linalg.norm(query) + 1e-8)
    else:
        raise ValueError(f"Unknown metric: {metric}")

    return np.argsort(-similarities)


if __name__ == "__main__":
    # 测试工具函数
    print("Testing Matrix Utility Functions")

    # 测试矩阵可逆性
    M = np.random.randn(5, 5)
    print(f"\nMatrix invertible: {is_invertible(M)}")

    # 测试正交矩阵生成
    Q = generate_orthogonal_matrix(5, seed=42)
    print(f"Q^T @ Q close to identity: {np.allclose(Q.T @ Q, np.eye(5))}")

    # 测试向量归一化
    vectors = np.random.randn(10, 5)
    normalized = normalize_vectors(vectors)
    norms = np.linalg.norm(normalized, axis=1)
    print(f"\nNormalized vectors have unit norm: {np.allclose(norms, 1.0)}")

    # 测试k-NN搜索
    query = np.random.randn(3, 5)
    database = np.random.randn(100, 5)
    indices, distances = find_k_nearest_neighbors(query, database, k=5)
    print(f"\nk-NN indices shape: {indices.shape}")
    print(f"k-NN distances shape: {distances.shape}")
