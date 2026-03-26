"""
检索服务抽象基类

定义检索服务的统一接口，支持 DCMH 跨模态检索和 CIR 图像检索。
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
import numpy as np

from app.schemas.unified import (
    SearchMode,
    EncryptionMode,
    BaseSearchResult,
    TagToImageResult,
    ImageToTagResult,
    ImageToImageResult,
    HitStats,
    EncryptionInfo,
    ServiceStatus
)


@dataclass
class SearchContext:
    """
    检索上下文。

    包含检索所需的所有配置信息。
    """
    mode: SearchMode
    encryption: EncryptionMode
    dataset: str
    top_k: int = 10

    # 标签检索参数
    tag_indices: Optional[List[int]] = None

    # 图像检索参数
    query_image_data: Optional[bytes] = None  # 图像字节数据
    query_image_base64: Optional[str] = None  # Base64 编码

    # 查询哈希码/特征（内部使用）
    query_vector: Optional[np.ndarray] = None


class BaseSearchService(ABC):
    """
    检索服务抽象基类。

    定义统一的检索服务接口，支持：
    - 服务初始化
    - 多种检索模式
    - 明文/加密检索
    - 状态查询
    """

    @abstractmethod
    async def initialize(self, dataset: str) -> bool:
        """
        初始化服务。

        加载模型、构建/加载索引、初始化加密服务等。

        参数：
            dataset: 数据集名称

        返回：
            是否初始化成功
        """
        pass

    @abstractmethod
    async def search_by_tags(self,
                             tag_indices: List[int],
                             context: SearchContext) -> Tuple[List[TagToImageResult], Dict[str, Any]]:
        """
        标签→图像检索。

        参数：
            tag_indices: 标签索引列表
            context: 检索上下文

        返回：
            (检索结果列表, 元数据字典)
        """
        pass

    @abstractmethod
    async def search_by_image(self,
                              image_data: bytes,
                              context: SearchContext) -> Tuple[List[BaseSearchResult], Dict[str, Any]]:
        """
        图像检索。

        根据 context.mode 决定是 image_to_tag 还是 image_to_image。

        参数：
            image_data: 图像字节数据
            context: 检索上下文

        返回：
            (检索结果列表, 元数据字典)
        """
        pass

    @abstractmethod
    def get_status(self, dataset: str) -> ServiceStatus:
        """
        获取服务状态。

        参数：
            dataset: 数据集名称

        返回：
            服务状态对象
        """
        pass

    @abstractmethod
    def get_supported_datasets(self) -> List[str]:
        """
        获取支持的数据集列表。

        返回：
            支持的数据集名称列表
        """
        pass

    @abstractmethod
    def is_initialized(self, dataset: str) -> bool:
        """
        检查数据集是否已初始化。

        参数：
            dataset: 数据集名称

        返回：
            是否已初始化
        """
        pass

    @abstractmethod
    async def rebuild_index(self, dataset: str) -> bool:
        """
        重建索引。

        参数：
            dataset: 数据集名称

        返回：
            是否重建成功
        """
        pass

    def get_service_type(self) -> str:
        """
        获取服务类型。

        返回：
            服务类型字符串（dcmh 或 cir）
        """
        return "base"

    def _build_encryption_info(self,
                               context: SearchContext,
                               query_encrypted: bool = False,
                               database_encrypted: bool = False) -> EncryptionInfo:
        """
        构建加密信息对象。

        参数：
            context: 检索上下文
            query_encrypted: 查询是否加密
            database_encrypted: 数据库是否加密

        返回：
            EncryptionInfo 对象
        """
        return EncryptionInfo(
            method="ASPE",
            query_encrypted=query_encrypted,
            database_encrypted=database_encrypted,
            security_level=3,
            bit_dim=self._get_bit_dim() if self.get_service_type() == "dcmh" else None,
            feature_dim=self._get_feature_dim() if self.get_service_type() == "cir" else None
        )

    def _get_bit_dim(self) -> Optional[int]:
        """获取哈希码位数（DCMH）。"""
        return None

    def _get_feature_dim(self) -> Optional[int]:
        """获取特征维度（CIR）。"""
        return None


class SearchServiceFactory:
    """
    检索服务工厂。

    根据数据集名称创建对应的检索服务实例。
    """

    _instances: Dict[str, BaseSearchService] = {}

    @classmethod
    def get_service(cls, dataset: str) -> Optional[BaseSearchService]:
        """
        获取检索服务实例。

        参数：
            dataset: 数据集名称

        返回：
            检索服务实例，如果数据集不支持则返回 None
        """
        from app.schemas.unified import DCMH_DATASETS, CIR_DATASETS

        if dataset in DCMH_DATASETS:
            from app.services.dcmh_search_service import get_dcmh_search_service
            return get_dcmh_search_service()
        elif dataset in CIR_DATASETS:
            from app.services.cir_search_service import get_cir_search_service
            return get_cir_search_service()

        return None

    @classmethod
    def register_service(cls, dataset: str, service: BaseSearchService) -> None:
        """
        注册服务实例。

        参数：
            dataset: 数据集名称
            service: 服务实例
        """
        cls._instances[dataset] = service

    @classmethod
    def clear_instances(cls) -> None:
        """清除所有服务实例（用于测试）。"""
        cls._instances.clear()