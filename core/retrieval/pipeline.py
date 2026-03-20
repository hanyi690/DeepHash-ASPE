"""
端到端检索管道

结合深度哈希模型 + ASPE 加密 + 安全检索，
实现完整的隐私保护跨模态检索。
"""

import torch
import numpy as np
from typing import List, Dict, Union, Optional, Tuple
from PIL import Image

from core.hashing.dual_stream import DualStreamHashModel
from core.retrieval.secure_retrieval import SecureRetrievalEngine


class RetrievalPipeline:
    """
    端到端隐私保护检索管道。

    结合了：
    1. 深度哈希模型（特征提取）
    2. ASPE 加密（隐私保护）
    3. 安全检索（密文内积）

    支持所有检索模式：
    - 文本 → 图像
    - 图像 → 文本
    - 图像 → 图像
    - 文本 → 文本
    """

    def __init__(self,
                 hash_model: DualStreamHashModel,
                 aspe_scheme: str = 'scheme1',
                 device: str = 'cpu'):
        """
        初始化检索管道。

        参数：
            hash_model: 训练好的双流哈希模型
            aspe_scheme: 'scheme1' 或 'scheme2'
            device: 'cpu' 或 'cuda'
        """
        self.device = device
        self.hash_model = hash_model.to(device)
        self.hash_model.eval()

        self.feature_dim = hash_model.feature_dim

        self.engine = SecureRetrievalEngine(
            aspe_scheme=aspe_scheme,
            feature_dim=self.feature_dim
        )

    def prepare_database(self,
                        images: Optional[List] = None,
                        texts: Optional[List] = None,
                        image_metadata: Optional[List] = None,
                        text_metadata: Optional[List] = None,
                        batch_size: int = 32) -> Dict:
        """
        提取特征并构建加密数据库。

        参数：
            images: PIL 图像或图像张量的列表
            texts: 文本字符串或标记索引的列表
            image_metadata: 图像的元数据
            text_metadata: 文本的元数据
            batch_size: 特征提取的批大小

        返回：
            包含处理统计信息的字典
        """
        stats = {
            'num_images': 0,
            'num_texts': 0,
            'image_extraction_time': 0,
            'text_extraction_time': 0,
            'encryption_time': 0
        }

        import time

        # 处理图像
        if images is not None and len(images) > 0:
            print(f"正在为 {len(images)} 张图像提取特征...")
            start_time = time.time()

            image_features = self._extract_image_features(images, batch_size)

            stats['num_images'] = len(images)
            stats['image_extraction_time'] = time.time() - start_time

        else:
            image_features = None

        # 处理文本
        if texts is not None and len(texts) > 0:
            print(f"正在为 {len(texts)} 个文本提取特征...")
            start_time = time.time()

            text_features = self._extract_text_features(texts, batch_size)

            stats['num_texts'] = len(texts)
            stats['text_extraction_time'] = time.time() - start_time

        else:
            text_features = None

        # 构建加密数据库
        print("正在构建加密数据库...")
        start_time = time.time()

        self.engine.build_encrypted_database(
            image_features=image_features,
            text_features=text_features,
            image_metadata=image_metadata,
            text_metadata=text_metadata
        )

        stats['encryption_time'] = time.time() - start_time

        print(f"数据库构建成功！")
        print(f"  图像数量: {stats['num_images']}")
        print(f"  文本数量: {stats['num_texts']}")
        print(f"  总耗时: {stats['image_extraction_time'] + stats['text_extraction_time'] + stats['encryption_time']:.2f}秒")

        return stats

    def _extract_image_features(self,
                                images: List,
                                batch_size: int) -> np.ndarray:
        """
        使用哈希模型从图像中提取特征。

        参数：
            images: PIL 图像或张量的列表
            batch_size: 批大小

        返回：
            [N, d] 形状的特征数组
        """
        features_list = []

        with torch.no_grad():
            for i in range(0, len(images), batch_size):
                batch = images[i:i + batch_size]

                # 如果需要，转换为张量
                if isinstance(batch[0], Image.Image):
                    # 假设预处理在其他地方完成
                    # 现在，将 PIL 图像转换为张量
                    batch_tensors = []
                    for img in batch:
                        # 调整大小为 224x224 并转换为张量
                        img = img.resize((224, 224))
                        img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
                        batch_tensors.append(img_tensor)
                    batch = torch.stack(batch_tensors)

                # 移动到设备
                batch = batch.to(self.device)

                # 提取特征
                batch_features = self.hash_model.get_features('image', batch)

                # 转换为 numpy
                features_list.append(batch_features.cpu().numpy())

        # 拼接所有批次
        features = np.concatenate(features_list, axis=0)
        return features

    def _extract_text_features(self,
                              texts: List,
                              batch_size: int) -> np.ndarray:
        """
        使用哈希模型从文本中提取特征。

        参数：
            texts: 标记索引或文本字符串的列表
            batch_size: 批大小

        返回：
            [N, d] 形状的特征数组
        """
        features_list = []

        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]

                # 如果需要，转换为张量
                if isinstance(batch[0], str):
                    # 文本字符串应该预先标记化
                    # 目前，假设它们已经被标记化
                    raise NotImplementedError("文本标记化未实现。请提供预先标记化的索引。")

                batch = torch.tensor(batch)
                mask = (batch == 0)  # 假设填充标记为 0

                # 移动到设备
                batch = batch.to(self.device)
                mask = mask.to(self.device)

                # 提取特征
                batch_features = self.hash_model.get_features('text', batch, mask)

                # 转换为 numpy
                features_list.append(batch_features.cpu().numpy())

        # 拼接所有批次
        features = np.concatenate(features_list, axis=0)
        return features

    def retrieve(self,
                query: Union[Image, np.ndarray, torch.Tensor, List],
                query_modality: str,
                target_modality: str,
                k: int = 10) -> List[Dict]:
        """
        完整的隐私保护检索。

        工作流程：
        1. 使用深度哈希模型提取查询特征
        2. 加密查询以创建陷门向量
        3. 通过密文内积执行安全检索
        4. 返回排序结果

        参数：
            query: 查询图像、文本或预先提取的特征
            query_modality: 'image' 或 'text'
            target_modality: 'image' 或 'text'
            k: 结果数量

        返回：
            带有元数据和分数的前 k 个结果
        """
        # 步骤 1：提取查询特征
        if isinstance(query, np.ndarray):
            query_feature = query
        elif isinstance(query, torch.Tensor):
            query_feature = query.cpu().numpy()
        else:
            # 使用哈希模型提取特征
            with torch.no_grad():
                if query_modality == 'image':
                    if isinstance(query, Image.Image):
                        # 预处理单张图像
                        query = query.resize((224, 224))
                        query_tensor = torch.from_numpy(np.array(query)).permute(2, 0, 1).float() / 255.0
                        query_tensor = query_tensor.unsqueeze(0)  # 添加批次维度
                    else:
                        query_tensor = query

                    query_tensor = query_tensor.to(self.device)
                    query_feature_tensor = self.hash_model.get_features('image', query_tensor)
                    query_feature = query_feature_tensor.cpu().numpy()[0]

                elif query_modality == 'text':
                    if isinstance(query, list):
                        query_tensor = torch.tensor(query).unsqueeze(0)
                    else:
                        query_tensor = query.unsqueeze(0)

                    mask = (query_tensor == 0)
                    query_tensor = query_tensor.to(self.device)
                    mask = mask.to(self.device)

                    query_feature_tensor = self.hash_model.get_features('text', query_tensor, mask)
                    query_feature = query_feature_tensor.cpu().numpy()[0]

                else:
                    raise ValueError(f"未知的 query_modality: {query_modality}")

        # 步骤 2-4：执行安全检索
        results = self.engine.query(
            query_feature=query_feature,
            query_modality=query_modality,
            target_modality=target_modality,
            k=k
        )

        return results

    # 不同检索模式的便捷方法
    def text_to_image(self, text_query, k: int = 10) -> List[Dict]:
        """文本查询 → 图像检索。"""
        return self.retrieve(text_query, 'text', 'image', k)

    def image_to_text(self, image_query, k: int = 10) -> List[Dict]:
        """图像查询 → 文本检索。"""
        return self.retrieve(image_query, 'image', 'text', k)

    def image_to_image(self, image_query, k: int = 10) -> List[Dict]:
        """图像查询 → 图像检索。"""
        return self.retrieve(image_query, 'image', 'image', k)

    def text_to_text(self, text_query, k: int = 10) -> List[Dict]:
        """文本查询 → 文本检索。"""
        return self.retrieve(text_query, 'text', 'text', k)

    def evaluate_retrieval(self,
                          query_features: np.ndarray,
                          query_labels: np.ndarray,
                          db_labels: np.ndarray,
                          query_modality: str,
                          target_modality: str,
                          k_values: List[int] = [1, 5, 10]) -> Dict:
        """
        评估检索性能。

        参数：
            query_features: [N, d] 查询特征
            query_labels: [N] 查询标签
            db_labels: [M] 数据库标签
            query_modality: 查询模态
            target_modality: 目标模态
            k_values: 精确度/召回率的 k 值

        返回：
            包含指标的字典
        """
        metrics = {}

        # 对所有查询执行检索
        all_results = self.engine.batch_query(
            query_features,
            query_modality,
            target_modality,
            k=max(k_values)
        )

        # 计算 precision@k 和 recall@k
        for k in k_values:
            precisions = []
            recalls = []

            for i, results in enumerate(all_results):
                # 获取前 k 个结果
                top_k = results[:k]
                retrieved_labels = [db_labels[r['rank'] - 1] for r in top_k]

                # 计算精确度
                precision = sum(1 for label in retrieved_labels if label == query_labels[i]) / k
                precisions.append(precision)

                # 计算召回率
                # 假设每个标签在数据库中至少有一个匹配
                relevant = sum(1 for label in db_labels if label == query_labels[i])
                recall = sum(1 for label in retrieved_labels if label == query_labels[i]) / max(relevant, 1)
                recalls.append(recall)

            metrics[f'precision@{k}'] = np.mean(precisions)
            metrics[f'recall@{k}'] = np.mean(recalls)

        return metrics

    def save_database(self, filepath: str):
        """
        将加密数据库保存到文件。

        参数：
            filepath: 保存数据库的路径
        """
        import pickle

        data = {
            'encrypted_db': self.engine.encrypted_db,
            'metadata': self.engine.metadata,
            'scheme': self.engine.scheme,
            'feature_dim': self.engine.feature_dim
        }

        with open(filepath, 'wb') as f:
            pickle.dump(data, f)

        print(f"数据库已保存到 {filepath}")

    def load_database(self, filepath: str):
        """
        从文件加载加密数据库。

        参数：
            filepath: 加载数据库的路径
        """
        import pickle

        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        self.engine.encrypted_db = data['encrypted_db']
        self.engine.metadata = data['metadata']

        print(f"数据库已从 {filepath} 加载")
        print(f"  图像数量: {len(self.engine.encrypted_db['image'])}")
        print(f"  文本数量: {len(self.engine.encrypted_db['text'])}")


if __name__ == "__main__":
    # 测试检索管道
    print("正在测试检索管道")

    # 创建虚拟哈希模型
    import torch
    from core.hashing.dual_stream import DualStreamHashModel

    vocab_size = 1000
    feature_dim = 256

    hash_model = DualStreamHashModel(
        vocab_size=vocab_size,
        feature_dim=feature_dim,
        hash_bits=32,
        image_backbone='resnet18',
        text_encoder_type='lstm',
        pretrained=False,
        text_num_layers=2
    )

    # 创建管道
    pipeline = RetrievalPipeline(
        hash_model=hash_model,
        aspe_scheme='scheme1',
        device='cpu'
    )

    # 生成虚拟数据
    num_images = 50
    num_texts = 30

    # 虚拟图像（随机张量）
    dummy_images = [torch.randn(3, 224, 224) for _ in range(num_images)]
    image_metadata = [f"image_{i}" for i in range(num_images)]

    # 虚拟文本（随机标记）
    dummy_texts = [torch.randint(0, vocab_size, (20,)) for _ in range(num_texts)]
    text_metadata = [f"text_{i}" for i in range(num_texts)]

    # 准备数据库
    print("\n=== 准备数据库 ===")
    stats = pipeline.prepare_database(
        images=dummy_images,
        texts=dummy_texts,
        image_metadata=image_metadata,
        text_metadata=text_metadata,
        batch_size=16
    )

    # 测试检索
    print("\n=== 测试检索 ===")

    # 文本 -> 图像
    query_text = torch.randint(0, vocab_size, (20,))
    results = pipeline.text_to_image(query_text, k=5)
    print(f"\n文本 -> 图像结果：")
    for r in results[:3]:
        print(f"  排名 {r['rank']}: {r['metadata']}, 分数={r['score']:.4f}")

    # 图像 -> 文本
    query_image = torch.randn(3, 224, 224)
    results = pipeline.image_to_text(query_image, k=5)
    print(f"\n图像 -> 文本结果：")
    for r in results[:3]:
        print(f"  排名 {r['rank']}: {r['metadata']}, 分数={r['score']:.4f}")

    print("\n管道测试完成！")
