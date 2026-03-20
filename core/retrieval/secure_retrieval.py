"""
隐私保护检索引擎

实现使用 ASPE 加密特征的安全相似性搜索。
通过密文内积支持所有检索模式。
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from core.aspe.scheme1 import ASPEScheme1
from core.aspe.scheme2 import ASPEScheme2


class SecureRetrievalEngine:
    """
    使用 ASPE 加密特征的隐私保护检索。

    通过密文内积支持所有检索模式：
    - 文本 → 图像
    - 图像 → 文本
    - 图像 → 图像
    - 文本 → 文本

    工作流程：
    1. 构建加密数据库：p_i = ET(feature_i)
    2. 查询：生成陷阱门 q_enc = EQ(query_feature)
    3. 排序：计算密文内积 score_i = p_i · q_enc
    4. 返回 top-k 结果
    """

    def __init__(self,
                 aspe_scheme: str = 'scheme1',
                 feature_dim: int = 4096,
                 seed: int = 42):
        """
        初始化安全检索引擎。

        参数：
            aspe_scheme: 'scheme1' 或 'scheme2'
            feature_dim: 特征向量维度
            seed: 密钥生成的随机种子
        """
        self.scheme = aspe_scheme
        self.feature_dim = feature_dim
        self.seed = seed

        # 初始化 ASPE
        if aspe_scheme == 'scheme1':
            self.aspe = ASPEScheme1(d=feature_dim, seed=seed)
        elif aspe_scheme == 'scheme2':
            self.aspe = ASPEScheme2(d=feature_dim, seed=seed)
        else:
            raise ValueError(f"未知的 ASPE 方案：{aspe_scheme}")

        # 加密数据库存储
        self.encrypted_db = {
            'image': [],  # 加密的图像特征
            'text': []    # 加密的文本特征
        }

        # 结果的元数据
        self.metadata = {
            'image': [],
            'text': []
        }

    def build_encrypted_database(self,
                                 image_features: Optional[np.ndarray] = None,
                                 text_features: Optional[np.ndarray] = None,
                                 image_metadata: Optional[List] = None,
                                 text_metadata: Optional[List] = None):
        """
        使用 ASPE 加密数据库特征。

        参数：
            image_features: 图像特征向量数组 [N_img, d]
            text_features: 文本特征向量数组 [N_txt, d]
            image_metadata: 每个图像的元数据列表
            text_metadata: 每个文本的元数据列表
        """
        # 加密图像特征
        if image_features is not None:
            if self.scheme == 'scheme1':
                self.encrypted_db['image'] = self.aspe.encrypt_database(image_features)
            else:  # scheme2
                self.encrypted_db['image'] = self.aspe.encrypt_database(image_features)

            self.metadata['image'] = image_metadata or [None] * len(image_features)
            print(f"已加密 {len(image_features)} 个图像特征")

        # 加密文本特征
        if text_features is not None:
            if self.scheme == 'scheme1':
                self.encrypted_db['text'] = self.aspe.encrypt_database(text_features)
            else:  # scheme2
                self.encrypted_db['text'] = self.aspe.encrypt_database(text_features)

            self.metadata['text'] = text_metadata or [None] * len(text_features)
            print(f"已加密 {len(text_features)} 个文本特征")

    def query(self,
              query_feature: np.ndarray,
              query_modality: str,
              target_modality: str,
              k: int = 10) -> List[Dict]:
        """
        使用密文内积进行隐私保护查询。

        工作流程：
        1. 生成陷阱门向量：q_enc = EQ(query_feature)
        2. 计算分数：score_i = ciphertext_inner_product(p_i, q_enc)
        3. 按分数排序并返回 top-k

        参数：
            query_feature: 查询特征向量 [d]
            query_modality: 'image' 或 'text'
            target_modality: 'image' 或 'text'
            k: 返回的结果数量

        返回：
            top-k 结果的 {metadata, score, rank} 列表
        """
        if query_modality not in ['image', 'text']:
            raise ValueError(f"无效的 query_modality：{query_modality}")
        if target_modality not in ['image', 'text']:
            raise ValueError(f"无效的 target_modality：{target_modality}")

        # 步骤 1：加密查询以创建陷阱门向量
        trapdoor = self.aspe.encrypt_query(query_feature)

        # 步骤 2：获取目标加密数据库
        if self.scheme == 'scheme1':
            target_db = self.encrypted_db[target_modality]
        else:  # scheme2
            target_db_a = self.encrypted_db[target_modality][0]
            target_db_b = self.encrypted_db[target_modality][1]

        target_metadata = self.metadata[target_modality]

        # 步骤 3：计算密文内积
        scores = []
        if self.scheme == 'scheme1':
            for idx, enc_feat in enumerate(target_db):
                score = self.aspe.ciphertext_inner_product(enc_feat, trapdoor)
                scores.append((score, idx))
        else:  # scheme2
            for idx in range(len(target_db_a)):
                enc_feat = (target_db_a[idx], target_db_b[idx])
                score = self.aspe.ciphertext_inner_product(enc_feat, trapdoor)
                scores.append((score, idx))

        # 步骤 4：排序并返回 top-k
        scores.sort(reverse=True, key=lambda x: x[0])
        top_k = scores[:k]

        results = []
        for score, idx in top_k:
            results.append({
                'metadata': target_metadata[idx],
                'score': float(score),
                'rank': len(results) + 1
            })

        return results

    def batch_query(self,
                   query_features: np.ndarray,
                   query_modality: str,
                   target_modality: str,
                   k: int = 10) -> List[List[Dict]]:
        """
        批量查询处理多个查询。

        参数：
            query_features: 查询特征数组 [N, d]
            query_modality: 'image' 或 'text'
            target_modality: 'image' 或 'text'
            k: 每个查询的结果数量

        返回：
            结果列表的列表，每个查询一个
        """
        results = []
        for query in query_features:
            result = self.query(query, query_modality, target_modality, k)
            results.append(result)
        return results

    def _ciphertext_inner_product(self,
                                  p_enc: Union[np.ndarray, Tuple],
                                  q_enc: Union[np.ndarray, Tuple]) -> float:
        """
        在密文空间计算内积。

        参数：
            p_enc: 加密的数据库点
            q_enc: 加密的查询点

        返回：
            内积值
        """
        if self.scheme == 'scheme1':
            return np.dot(p_enc[:-1], q_enc[:-1])
        else:  # scheme2
            p_a, p_b = p_enc
            q_a, q_b = q_enc
            return np.dot(p_a, q_a) + np.dot(p_b, q_b)

    # 不同检索模式的便捷方法
    def text_to_image(self, text_query: np.ndarray, k: int = 10) -> List[Dict]:
        """
        文本查询 → 图像检索。

        参数：
            text_query: 文本特征向量 [d]
            k: 结果数量

        返回：
            top-k 图像结果
        """
        return self.query(text_query, 'text', 'image', k)

    def image_to_text(self, image_query: np.ndarray, k: int = 10) -> List[Dict]:
        """
        图像查询 → 文本检索。

        参数：
            image_query: 图像特征向量 [d]
            k: 结果数量

        返回：
            top-k 文本结果
        """
        return self.query(image_query, 'image', 'text', k)

    def image_to_image(self, image_query: np.ndarray, k: int = 10) -> List[Dict]:
        """
        图像查询 → 图像检索。

        参数：
            image_query: 图像特征向量 [d]
            k: 结果数量

        返回：
            top-k 图像结果
        """
        return self.query(image_query, 'image', 'image', k)

    def text_to_text(self, text_query: np.ndarray, k: int = 10) -> List[Dict]:
        """
        文本查询 → 文本检索。

        参数：
            text_query: 文本特征向量 [d]
            k: 结果数量

        返回：
            top-k 文本结果
        """
        return self.query(text_query, 'text', 'text', k)

    def get_database_size(self) -> Dict[str, int]:
        """
        获取加密数据库的大小。

        返回：
            包含每种模态计数的字典
        """
        return {
            'image': len(self.encrypted_db['image']),
            'text': len(self.encrypted_db['text'])
        }

    def clear_database(self):
        """清除数据库中的所有加密数据。"""
        self.encrypted_db = {'image': [], 'text': []}
        self.metadata = {'image': [], 'text': []}

    def get_scheme_info(self) -> Dict:
        """
        获取正在使用的 ASPE 方案的信息。

        返回：
            包含方案信息的字典
        """
        info = {
            'scheme': self.scheme,
            'feature_dim': self.feature_dim,
            'seed': self.seed
        }

        if self.scheme == 'scheme2':
            info['d_prime'] = self.aspe.d_prime

        return info


class SecureKNN:
    """
    在加密数据上进行 K 近邻搜索。

    使用 ASPE 加密特征提供高效的 k-NN 搜索。
    """

    def __init__(self,
                 aspe_scheme: str = 'scheme1',
                 feature_dim: int = 4096,
                 seed: int = 42):
        """
        初始化安全 KNN。

        参数：
            aspe_scheme: 使用的 ASPE 方案
            feature_dim: 特征维度
            seed: 随机种子
        """
        self.engine = SecureRetrievalEngine(aspe_scheme, feature_dim, seed)
        self.scheme = aspe_scheme

    def fit(self,
            features: np.ndarray,
            modality: str,
            metadata: Optional[List] = None):
        """
        构建 k-NN 搜索的加密数据库。

        参数：
            features: 特征向量 [N, d]
            modality: 'image' 或 'text'
            metadata: 每个特征的可选元数据
        """
        if modality == 'image':
            self.engine.build_encrypted_database(
                image_features=features,
                image_metadata=metadata
            )
        else:
            self.engine.build_encrypted_database(
                text_features=features,
                text_metadata=metadata
            )

    def kneighbors(self,
                   query: np.ndarray,
                   k: int = 5,
                   modality: str = 'image') -> Tuple[np.ndarray, np.ndarray]:
        """
        查找查询的 k 个最近邻。

        参数：
            query: 查询特征向量 [d]
            k: 邻居数量
            modality: 要搜索的目标模态

        返回：
            (索引，距离/相似度) 的元组
        """
        # 查询引擎
        results = self.engine.query(
            query_feature=query,
            query_modality=modality,
            target_modality=modality,
            k=k
        )

        # 提取索引和分数
        indices = np.array([r['rank'] - 1 for r in results])
        scores = np.array([r['score'] for r in results])

        return indices, scores


if __name__ == "__main__":
    # 测试安全检索引擎
    print("正在测试安全检索引擎")

    # 参数
    feature_dim = 128  # 测试时使用较小的维度
    num_images = 100
    num_texts = 50

    # 生成虚拟特征
    image_features = np.random.randn(num_images, feature_dim)
    text_features = np.random.randn(num_texts, feature_dim)

    # 生成元数据
    image_metadata = [f"image_{i}" for i in range(num_images)]
    text_metadata = [f"text_{i}" for i in range(num_texts)]

    # 测试方案 1
    print("\n=== 测试方案 1 ===")
    engine1 = SecureRetrievalEngine(aspe_scheme='scheme1', feature_dim=feature_dim)

    # 构建数据库
    engine1.build_encrypted_database(
        image_features=image_features,
        text_features=text_features,
        image_metadata=image_metadata,
        text_metadata=text_metadata
    )

    # 测试查询
    query_img = np.random.randn(feature_dim)
    query_txt = np.random.randn(feature_dim)

    # 文本 -> 图像
    results_t2i = engine1.text_to_image(query_txt, k=5)
    print(f"\n文本 -> 图像检索：")
    for r in results_t2i[:3]:
        print(f"  排名 {r['rank']}: {r['metadata']}, 分数={r['score']:.4f}")

    # 图像 -> 文本
    results_i2t = engine1.image_to_text(query_img, k=5)
    print(f"\n图像 -> 文本检索：")
    for r in results_i2t[:3]:
        print(f"  排名 {r['rank']}: {r['metadata']}, 分数={r['score']:.4f}")

    # 测试方案 2
    print("\n=== 测试方案 2 ===")
    engine2 = SecureRetrievalEngine(aspe_scheme='scheme2', feature_dim=feature_dim)

    engine2.build_encrypted_database(
        image_features=image_features,
        text_features=text_features,
        image_metadata=image_metadata,
        text_metadata=text_metadata
    )

    # 文本 -> 图像
    results_t2i = engine2.text_to_image(query_txt, k=5)
    print(f"\n文本 -> 图像检索：")
    for r in results_t2i[:3]:
        print(f"  排名 {r['rank']}: {r['metadata']}, 分数={r['score']:.4f}")

    # 获取方案信息
    print(f"\n方案 1 信息：{engine1.get_scheme_info()}")
    print(f"方案 2 信息：{engine2.get_scheme_info()}")
