"""
DCMH 跨模态检索服务

提供基于深度哈希的跨模态检索功能：
- 标签搜图（tag_to_image）
- 图搜标签（image_to_tag）

支持明文和加密检索。
"""

import numpy as np
import torch
from typing import Dict, Any, Optional, List, Tuple
import time
import logging
from pathlib import Path

from app.services.base_search_service import BaseSearchService, SearchContext
from app.schemas.unified import (
    SearchMode,
    EncryptionMode,
    BaseSearchResult,
    TagToImageResult,
    ImageToTagResult,
    HitStats,
    EncryptionInfo,
    ServiceStatus,
    DCMH_DATASETS
)
from app.services.dcmh_encryption_service import get_dcmh_encryption_service, DCMHEncryptionService
from app.services.dcmh_service import get_dcmh_service, DCMHService
from app.services.dataset_service import get_dataset_service, DATASET_CONFIGS
from app.services.hash_cache_service import get_hash_cache_service

logger = logging.getLogger(__name__)


class DCMHSearchService(BaseSearchService):
    """
    DCMH 跨模态检索服务。

    支持的检索模式：
    - tag_to_image: 标签搜图
    - image_to_tag: 图搜标签

    支持的数据集：
    - flickr25k
    - nuswide
    """

    def __init__(self, bit_dim: int = 64):
        """
        初始化 DCMH 检索服务。

        参数：
            bit_dim: 哈希码位数
        """
        self.bit_dim = bit_dim

        # 服务组件（延迟初始化）
        self._encryption_service: Optional[DCMHEncryptionService] = None

        # 按数据集缓存
        self._initialized_datasets: Dict[str, bool] = {}
        self._image_codes: Dict[str, np.ndarray] = {}
        self._text_codes: Dict[str, np.ndarray] = {}
        self._encrypted_image_codes: Dict[str, np.ndarray] = {}
        self._encrypted_text_codes: Dict[str, np.ndarray] = {}
        self._yall: Dict[str, np.ndarray] = {}
        self._lall: Dict[str, np.ndarray] = {}

        logger.info(f"[DCMHSearchService] 初始化：bit_dim={bit_dim}")

    def get_service_type(self) -> str:
        """获取服务类型。"""
        return "dcmh"

    def _get_bit_dim(self) -> Optional[int]:
        """获取哈希码位数。"""
        return self.bit_dim

    @property
    def encryption_service(self) -> DCMHEncryptionService:
        """获取加密服务（延迟初始化）。"""
        if self._encryption_service is None:
            self._encryption_service = get_dcmh_encryption_service(bit_dim=self.bit_dim)
        return self._encryption_service

    async def initialize(self, dataset: str) -> bool:
        """
        初始化数据集。

        参数：
            dataset: 数据集名称

        返回：
            是否初始化成功
        """
        if dataset in self._initialized_datasets:
            return True

        try:
            hash_cache = get_hash_cache_service()
            dcmh_service = get_dcmh_service(dataset=dataset)
            dataset_service = get_dataset_service(dataset_name=dataset)

            # 检查模型是否已加载
            if not dcmh_service.is_loaded():
                logger.warning(f"[DCMHSearchService] DCMH 模型未加载 ({dataset})")

            # 加载哈希码缓存
            image_codes, text_codes, yall = hash_cache.load_dcmh_yall(dataset)

            if image_codes is not None:
                self._image_codes[dataset] = image_codes
                self._text_codes[dataset] = text_codes
                self._yall[dataset] = yall

                # 加载 LAll 类别标签
                lall = hash_cache.load_dcmh_lall(dataset)
                if lall is not None:
                    self._lall[dataset] = lall

                # 加载或生成加密缓存
                encrypted_image = hash_cache.load_dcmh_encrypted(dataset)
                if encrypted_image is not None:
                    self._encrypted_image_codes[dataset] = encrypted_image
                else:
                    # 加密图像哈希码
                    encrypted_image = self.encryption_service.encrypt_database(image_codes)
                    self._encrypted_image_codes[dataset] = encrypted_image
                    hash_cache.save_dcmh_encrypted(encrypted_image, dataset)

                encrypted_text = hash_cache.load_dcmh_encrypted_text(dataset)
                if encrypted_text is not None:
                    self._encrypted_text_codes[dataset] = encrypted_text
                else:
                    # 加密文本哈希码
                    encrypted_text = self.encryption_service.encrypt_text_database(text_codes)
                    self._encrypted_text_codes[dataset] = encrypted_text
                    hash_cache.save_dcmh_encrypted_text(encrypted_text, dataset)

                self._initialized_datasets[dataset] = True
                logger.info(f"[DCMHSearchService] 数据集初始化完成 ({dataset})：图像 {image_codes.shape[0]} 条")
                return True
            else:
                logger.error(f"[DCMHSearchService] 无法加载缓存 ({dataset})")
                return False

        except Exception as e:
            logger.error(f"[DCMHSearchService] 初始化失败 ({dataset})：{e}")
            return False

    async def search_by_tags(self,
                             tag_indices: List[int],
                             context: SearchContext) -> Tuple[List[TagToImageResult], Dict[str, Any]]:
        """
        标签搜图。

        参数：
            tag_indices: 标签索引列表
            context: 检索上下文

        返回：
            (检索结果列表, 元数据字典)
        """
        start_time = time.time()

        dataset = context.dataset
        await self.initialize(dataset)

        dcmh_service = get_dcmh_service(dataset=dataset)
        dataset_service = get_dataset_service(dataset_name=dataset)
        hash_cache = get_hash_cache_service()

        # 生成查询哈希码
        tag_dim = 1386  # Flickr25K 标签维度
        tag_vector = np.zeros(tag_dim, dtype=np.float32)
        for idx in tag_indices:
            if 0 <= idx < tag_dim:
                tag_vector[idx] = 1.0

        text_tensor = torch.from_numpy(tag_vector).unsqueeze(0).float()
        query_code = dcmh_service.generate_text_code_single(text_tensor)
        query_code_np = query_code.cpu().numpy().squeeze()

        # 执行检索
        image_codes = self._image_codes.get(dataset)
        encrypted_image_codes = self._encrypted_image_codes.get(dataset)

        if context.encryption == EncryptionMode.ENCRYPTED and encrypted_image_codes is not None:
            # 加密检索
            encrypted_query = self.encryption_service.encrypt_query(query_code_np)
            distances = self.encryption_service.compute_distance(encrypted_query, encrypted_image_codes).squeeze()
            query_encrypted = True
            database_encrypted = True
        else:
            # 明文检索
            inner_products = np.dot(image_codes, query_code_np)
            distances = 0.5 * (self.bit_dim - inner_products)
            query_encrypted = False
            database_encrypted = False

        # 排序
        distances_rounded = np.round(distances, decimals=10)
        sorted_indices = np.lexsort((np.arange(len(distances_rounded)), distances_rounded))
        top_k_indices = sorted_indices[:context.top_k]

        # 获取检索库索引
        dataset_service.load_data()
        _, _, retrieval_indices = dataset_service.get_data_split_indices()
        retrieval_tags = dataset_service.get_yall(retrieval_indices)
        retrieval_lall = self._lall.get(dataset)

        # 构建结果
        results: List[TagToImageResult] = []
        query_yall_indices = set(tag_indices)
        query_tag_names = dataset_service.get_tag_names_from_yall_indices(list(query_yall_indices))

        # 计算查询 LAll 向量
        query_lall_vector = None
        if retrieval_lall is not None and retrieval_tags is not None:
            mask = np.zeros(len(retrieval_tags), dtype=bool)
            for tag_idx in query_yall_indices:
                if tag_idx < retrieval_tags.shape[1]:
                    mask |= (retrieval_tags[:, tag_idx] > 0)
            matching_lall = retrieval_lall[mask]
            if len(matching_lall) > 0:
                query_lall_vector = matching_lall.max(axis=0)

        total_tag_hits = 0
        total_category_hits = 0

        for rank, idx in enumerate(top_k_indices):
            actual_idx = retrieval_indices[idx] if idx < len(retrieval_indices) else idx
            image_id = int(actual_idx)

            # 获取标签
            image_tags = []
            if retrieval_tags is not None and idx < len(retrieval_tags):
                image_tags = np.where(retrieval_tags[idx] > 0)[0].tolist()

            hit_tags = [t for t in image_tags if t in query_yall_indices]
            tag_hit = len(hit_tags) > 0

            # 获取类别
            category_hit = False
            hit_category_names = []
            result_category_names = []
            if retrieval_lall is not None and query_lall_vector is not None and idx < len(retrieval_lall):
                result_lall = retrieval_lall[idx]
                category_hit = np.any((result_lall > 0) & (query_lall_vector > 0))

                result_category_indices = np.where(result_lall > 0)[0].tolist()
                result_category_names = dataset_service.get_category_names_from_lall_indices(result_category_indices)

                if category_hit:
                    hit_category_indices = np.where((result_lall > 0) & (query_lall_vector > 0))[0].tolist()
                    hit_category_names = dataset_service.get_category_names_from_lall_indices(hit_category_indices)

            tag_names = dataset_service.get_tag_names_from_yall_indices(image_tags[:20])
            hit_tag_names = dataset_service.get_tag_names_from_yall_indices(hit_tags)

            if tag_hit:
                total_tag_hits += 1
            if category_hit:
                total_category_hits += 1

            # 生成缩略图 URL
            config = DATASET_CONFIGS.get(dataset, {})
            if config.get('type') == 'raw':
                original_id = dataset_service.get_original_image_id(image_id)
                thumbnail_url = f"/flickr-images/im{original_id + 1}.jpg"
            else:
                thumbnail_url = f"/api/images/{image_id}?format=image&dataset={dataset}"

            distance = max(0.0, float(distances[idx]))
            score = float(1.0 / (1.0 + distance))

            results.append(TagToImageResult(
                rank=rank + 1,
                image_id=str(image_id),
                score=score,
                distance=distance,
                thumbnail_url=thumbnail_url,
                tags=image_tags[:20],
                tag_names=tag_names,
                hit_tags=hit_tags,
                hit_tag_names=hit_tag_names,
                category_hit=category_hit,
                tag_hit=tag_hit,
                category_names=result_category_names,
                hit_category_names=hit_category_names
            ))

        search_time_ms = (time.time() - start_time) * 1000

        hit_stats = {
            "total_results": len(results),
            "tag_hits": total_tag_hits,
            "tag_hit_rate": total_tag_hits / len(results) if results else 0,
            "category_hits": total_category_hits,
            "category_hit_rate": total_category_hits / len(results) if results else 0,
            "query_tags": list(query_yall_indices),
            "query_tag_names": query_tag_names
        }

        metadata = {
            "search_time_ms": search_time_ms,
            "hit_stats": hit_stats,
            "encryption_info": self._build_encryption_info(
                context, query_encrypted, database_encrypted
            ),
            "query_hash_code": query_code_np.tolist()
        }

        return results, metadata

    async def search_by_image(self,
                              image_data: bytes,
                              context: SearchContext) -> Tuple[List[BaseSearchResult], Dict[str, Any]]:
        """
        图像检索（图搜标签）。

        参数：
            image_data: 图像字节数据
            context: 检索上下文

        返回：
            (检索结果列表, 元数据字典)
        """
        start_time = time.time()

        dataset = context.dataset
        await self.initialize(dataset)

        dcmh_service = get_dcmh_service(dataset=dataset)
        dataset_service = get_dataset_service(dataset_name=dataset)
        hash_cache = get_hash_cache_service()

        # 处理图像并生成哈希码
        from PIL import Image
        from io import BytesIO
        from app.services.dcmh_service import preprocess_image_for_inference

        pil_image = Image.open(BytesIO(image_data))
        image_tensor = preprocess_image_for_inference(pil_image).unsqueeze(0)
        query_code = dcmh_service.generate_image_code_single(image_tensor)
        query_code_np = query_code.cpu().numpy().squeeze()

        # 执行检索
        text_codes = self._text_codes.get(dataset)
        encrypted_text_codes = self._encrypted_text_codes.get(dataset)

        if context.encryption == EncryptionMode.ENCRYPTED and encrypted_text_codes is not None:
            encrypted_query = self.encryption_service.encrypt_query(query_code_np)
            distances = self.encryption_service.compute_distance(encrypted_query, encrypted_text_codes).squeeze()
            query_encrypted = True
            database_encrypted = True
        else:
            inner_products = np.dot(text_codes, query_code_np)
            distances = 0.5 * (self.bit_dim - inner_products)
            query_encrypted = False
            database_encrypted = False

        # 排序
        distances_rounded = np.round(distances, decimals=10)
        sorted_indices = np.lexsort((np.arange(len(distances_rounded)), distances_rounded))
        top_k_indices = sorted_indices[:context.top_k]

        # 获取检索库索引
        dataset_service.load_data()
        _, _, retrieval_indices = dataset_service.get_data_split_indices()
        retrieval_tags = dataset_service.get_yall(retrieval_indices)
        retrieval_lall = self._lall.get(dataset)

        # 构建结果
        results: List[ImageToTagResult] = []
        category_counter = {}

        for rank, idx in enumerate(top_k_indices):
            actual_idx = retrieval_indices[idx] if idx < len(retrieval_indices) else idx
            image_id = int(actual_idx)

            image_tags = []
            if retrieval_tags is not None and idx < len(retrieval_tags):
                image_tags = np.where(retrieval_tags[idx] > 0)[0].tolist()

            tag_names = dataset_service.get_tag_names_from_yall_indices(image_tags[:20])

            result_category_names = []
            if retrieval_lall is not None and idx < len(retrieval_lall):
                result_lall = retrieval_lall[idx]
                result_category_indices = np.where(result_lall > 0)[0].tolist()
                result_category_names = dataset_service.get_category_names_from_lall_indices(result_category_indices)

                for cat in result_category_names:
                    category_counter[cat] = category_counter.get(cat, 0) + 1

            config = DATASET_CONFIGS.get(dataset, {})
            if config.get('type') == 'raw':
                original_id = dataset_service.get_original_image_id(image_id)
                thumbnail_url = f"/flickr-images/im{original_id + 1}.jpg"
            else:
                thumbnail_url = f"/api/images/{image_id}?format=image&dataset={dataset}"

            results.append(ImageToTagResult(
                rank=rank + 1,
                image_id=str(image_id),
                tags=image_tags[:20],
                tag_names=tag_names,
                score=float(1.0 / (1.0 + distances[idx])),
                distance=float(distances[idx]),
                thumbnail_url=thumbnail_url,
                category_names=result_category_names
            ))

        search_time_ms = (time.time() - start_time) * 1000

        hit_stats = {
            "total_results": len(results),
            "category_distribution": category_counter,
            "query_type": "image_to_tag"
        }

        metadata = {
            "search_time_ms": search_time_ms,
            "hit_stats": hit_stats,
            "encryption_info": self._build_encryption_info(
                context, query_encrypted, database_encrypted
            ),
            "query_hash_code": query_code_np.tolist()
        }

        return results, metadata

    def get_status(self, dataset: str) -> ServiceStatus:
        """获取服务状态。"""
        dcmh_service = get_dcmh_service(dataset=dataset)

        return ServiceStatus(
            success=True,
            service_type="dcmh",
            dataset=dataset,
            initialized=self._initialized_datasets.get(dataset, False),
            model_loaded=dcmh_service.is_loaded(),
            plaintext_indexed=dataset in self._image_codes,
            encrypted_indexed=dataset in self._encrypted_image_codes,
            index_size=len(self._image_codes.get(dataset, [])),
            keys_loaded=self.encryption_service.has_keys(),
            additional_info={
                "bit_dim": self.bit_dim,
                "y_dim": dcmh_service.y_dim
            }
        )

    def get_supported_datasets(self) -> List[str]:
        """获取支持的数据集列表。"""
        return DCMH_DATASETS

    def is_initialized(self, dataset: str) -> bool:
        """检查数据集是否已初始化。"""
        return self._initialized_datasets.get(dataset, False)

    async def rebuild_index(self, dataset: str) -> bool:
        """重建索引。"""
        self._initialized_datasets.pop(dataset, None)
        return await self.initialize(dataset)


# 全局服务实例
_dcmh_search_service: Optional[DCMHSearchService] = None


def get_dcmh_search_service(bit_dim: int = 64) -> DCMHSearchService:
    """获取 DCMH 检索服务实例。"""
    global _dcmh_search_service
    if _dcmh_search_service is None:
        _dcmh_search_service = DCMHSearchService(bit_dim=bit_dim)
    return _dcmh_search_service