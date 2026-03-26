"""
CIR CNN 图像检索服务

提供基于 CNN 特征的图像检索功能：
- 图搜图（image_to_image）

支持明文和加密检索。
"""

import numpy as np
import torch
from typing import Dict, Any, Optional, List, Tuple
import time
import logging
from pathlib import Path
import tempfile
import os

from app.services.base_search_service import BaseSearchService, SearchContext
from app.schemas.unified import (
    SearchMode,
    EncryptionMode,
    BaseSearchResult,
    ImageToImageResult,
    HitStats,
    EncryptionInfo,
    ServiceStatus,
    CIR_DATASETS
)
from app.services.cir_encryption_service import get_cir_encryption_service, CIREncryptionService
from app.services.cir_service import get_cir_service, CIRService
from app.services.hash_cache_service import get_hash_cache_service

logger = logging.getLogger(__name__)


class CIRSearchService(BaseSearchService):
    """
    CIR CNN 图像检索服务。

    支持的检索模式：
    - image_to_image: 图搜图

    支持的数据集：
    - roxford5k
    - rparis6k
    """

    def __init__(self, feature_dim: int = 2048):
        """
        初始化 CIR 检索服务。

        参数：
            feature_dim: 特征维度（默认 2048）
        """
        self.feature_dim = feature_dim

        # 服务组件（延迟初始化）
        self._encryption_service: Optional[CIREncryptionService] = None

        # 按数据集缓存
        self._initialized_datasets: Dict[str, bool] = {}
        self._plaintext_features: Dict[str, torch.Tensor] = {}
        self._encrypted_features: Dict[str, torch.Tensor] = {}
        self._image_names: Dict[str, List[str]] = {}

        logger.info(f"[CIRSearchService] 初始化：feature_dim={feature_dim}")

    def get_service_type(self) -> str:
        """获取服务类型。"""
        return "cir"

    def _get_feature_dim(self) -> Optional[int]:
        """获取特征维度。"""
        return self.feature_dim

    @property
    def encryption_service(self) -> CIREncryptionService:
        """获取加密服务（延迟初始化）。"""
        if self._encryption_service is None:
            self._encryption_service = get_cir_encryption_service(feature_dim=self.feature_dim)
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
            cir_service = get_cir_service()
            hash_cache = get_hash_cache_service()

            # 确保模型已加载
            if cir_service.model is None:
                model_path = Path("data/networks/gl18-tl-resnet101-gem-w-a4d43db.pth")
                if not model_path.exists():
                    model_path = Path(__file__).parent.parent.parent.parent / "data" / "networks" / "gl18-tl-resnet101-gem-w-a4d43db.pth"

                if model_path.exists():
                    logger.info(f"[CIRSearchService] 加载模型: {model_path}")
                    cir_service.load_model(str(model_path))
                else:
                    logger.error(f"[CIRSearchService] 模型不存在: {model_path}")
                    return False

            # 尝试从缓存加载
            if hash_cache.cir_cache_exists(dataset, "features"):
                features, image_names = hash_cache.load_cir_features(dataset)

                if features is not None and image_names is not None:
                    self._plaintext_features[dataset] = torch.from_numpy(features)
                    self._image_names[dataset] = image_names

                    # 加载加密缓存
                    encrypted = hash_cache.load_cir_encrypted(dataset)
                    if encrypted is not None:
                        self._encrypted_features[dataset] = torch.from_numpy(encrypted)
                    else:
                        logger.warning(f"[CIRSearchService] 加密缓存不存在 ({dataset})，仅支持明文检索")

                    self._initialized_datasets[dataset] = True
                    logger.info(f"[CIRSearchService] 数据集初始化完成 ({dataset})：{len(image_names)} 张图像")
                    return True

            logger.error(f"[CIRSearchService] 缓存不存在 ({dataset})")
            return False

        except Exception as e:
            logger.error(f"[CIRSearchService] 初始化失败 ({dataset})：{e}")
            return False

    async def search_by_tags(self,
                             tag_indices: List[int],
                             context: SearchContext) -> Tuple[List[BaseSearchResult], Dict[str, Any]]:
        """
        标签搜图（CIR 不支持）。

        参数：
            tag_indices: 标签索引列表
            context: 检索上下文

        返回：
            (空列表, 错误信息)
        """
        return [], {
            "error": "CIR 服务不支持标签搜图，请使用 image_to_image 模式"
        }

    async def search_by_image(self,
                              image_data: bytes,
                              context: SearchContext) -> Tuple[List[BaseSearchResult], Dict[str, Any]]:
        """
        图像检索（图搜图）。

        参数：
            image_data: 图像字节数据
            context: 检索上下文

        返回：
            (检索结果列表, 元数据字典)
        """
        start_time = time.time()

        dataset = context.dataset
        await self.initialize(dataset)

        cir_service = get_cir_service()

        # 处理图像并提取特征
        from PIL import Image
        from io import BytesIO

        pil_image = Image.open(BytesIO(image_data))
        pil_image = pil_image.convert('RGB')

        # 保存到临时文件
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            pil_image.save(tmp, format='JPEG')
            tmp_path = tmp.name

        try:
            # 提取查询特征
            query_features = cir_service.extract_features([tmp_path])
            query_feature = query_features[:, 0]  # [d]

            # L2 归一化
            query_feature = query_feature / (query_feature.norm() + 1e-10)
        finally:
            os.unlink(tmp_path)

        # 执行检索
        plaintext_features = self._plaintext_features.get(dataset)
        encrypted_features = self._encrypted_features.get(dataset)
        image_names = self._image_names.get(dataset)

        if context.encryption == EncryptionMode.ENCRYPTED and encrypted_features is not None:
            # 加密检索
            encrypted_query = self.encryption_service.encrypt_query(query_feature.cpu().numpy())
            scores = self.encryption_service.compute_similarity(
                encrypted_query, encrypted_features.cpu().numpy()
            ).squeeze()
            query_encrypted = True
            database_encrypted = True
        else:
            # 明文检索
            db_norm = plaintext_features / (
                plaintext_features.norm(dim=1, keepdim=True) + 1e-10
            )
            scores = torch.matmul(db_norm, query_feature).cpu().numpy()
            query_encrypted = False
            database_encrypted = False

        # Top-K 排序
        top_k = min(context.top_k, len(image_names))
        topk_indices = np.argsort(scores)[::-1][:top_k]  # 降序

        # 构建结果
        results: List[ImageToImageResult] = []

        for rank, idx in enumerate(topk_indices):
            image_name = image_names[idx]
            score = float(scores[idx])

            results.append(ImageToImageResult(
                rank=rank + 1,
                image_id=str(idx),
                score=score,
                distance=-score,  # 距离为负分数
                thumbnail_url=f"/cir-images/{dataset}/jpg/{image_name}",
                image_name=image_name,
                image_url=f"/cir-images/{dataset}/jpg/{image_name}"
            ))

        search_time_ms = (time.time() - start_time) * 1000

        hit_stats = {
            "total_results": len(results)
        }

        metadata = {
            "search_time_ms": search_time_ms,
            "hit_stats": hit_stats,
            "encryption_info": self._build_encryption_info(
                context, query_encrypted, database_encrypted
            )
        }

        return results, metadata

    def get_status(self, dataset: str) -> ServiceStatus:
        """获取服务状态。"""
        cir_service = get_cir_service()

        return ServiceStatus(
            success=True,
            service_type="cir",
            dataset=dataset,
            initialized=self._initialized_datasets.get(dataset, False),
            model_loaded=cir_service.model is not None,
            plaintext_indexed=dataset in self._plaintext_features,
            encrypted_indexed=dataset in self._encrypted_features,
            index_size=len(self._image_names.get(dataset, [])),
            keys_loaded=self.encryption_service.has_keys(),
            additional_info={
                "feature_dim": self.feature_dim
            }
        )

    def get_supported_datasets(self) -> List[str]:
        """获取支持的数据集列表。"""
        return CIR_DATASETS

    def is_initialized(self, dataset: str) -> bool:
        """检查数据集是否已初始化。"""
        return self._initialized_datasets.get(dataset, False)

    async def rebuild_index(self, dataset: str) -> bool:
        """重建索引。"""
        self._initialized_datasets.pop(dataset, None)
        return await self.initialize(dataset)


# 全局服务实例
_cir_search_service: Optional[CIRSearchService] = None


def get_cir_search_service(feature_dim: int = 2048) -> CIRSearchService:
    """获取 CIR 检索服务实例。"""
    global _cir_search_service
    if _cir_search_service is None:
        _cir_search_service = CIRSearchService(feature_dim=feature_dim)
    return _cir_search_service