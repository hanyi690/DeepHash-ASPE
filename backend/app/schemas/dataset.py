"""
数据集相关的 Pydantic 模式

定义数据集管理和状态查询的数据结构。
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


class DatasetInfo(BaseModel):
    """数据集基本信息。"""
    name: str = Field(..., description="数据集名称")
    display_name: str = Field(..., description="显示名称")
    type: str = Field(..., description="数据集类型：dcmh | cir")
    status: str = Field(default="unknown", description="状态：ready | loading | error | unknown")
    cache_exists: bool = Field(default=False, description="缓存是否存在")
    image_count: int = Field(default=0, description="图像数量")


class DatasetListResponse(BaseModel):
    """数据集列表响应。"""
    success: bool = True
    datasets: List[DatasetInfo] = []
    message: str = "数据集列表获取成功"


class DatasetStatus(BaseModel):
    """数据集状态详情。"""
    name: str = Field(..., description="数据集名称")
    cache_loaded: bool = Field(default=False, description="缓存是否已加载到内存")
    cache_path: str = Field(default="", description="缓存文件路径")
    model_loaded: bool = Field(default=False, description="模型是否已加载")
    gpu_available: bool = Field(default=False, description="GPU 是否可用")
    gpu_name: Optional[str] = Field(default=None, description="GPU 名称")
    database_size: int = Field(default=0, description="数据库大小")
    query_size: int = Field(default=0, description="查询集大小")
    bit_dim: int = Field(default=64, description="哈希码位数")


class DatasetStatusResponse(BaseModel):
    """数据集状态响应。"""
    success: bool = True
    status: DatasetStatus
    message: str = "数据集状态获取成功"


class SystemStatusResponse(BaseModel):
    """系统状态响应。"""
    success: bool = True
    gpu_available: bool = Field(default=False, description="GPU 是否可用")
    gpu_name: Optional[str] = Field(default=None, description="GPU 名称")
    device: str = Field(default="cpu", description="当前设备：cuda | cpu")
    datasets: Dict[str, DatasetStatus] = Field(default={}, description="各数据集状态")


class CacheBuildRequest(BaseModel):
    """缓存构建请求。"""
    dataset: str = Field(..., description="数据集名称")
    force_rebuild: bool = Field(default=False, description="是否强制重建")


class CacheBuildResponse(BaseModel):
    """缓存构建响应。"""
    success: bool
    dataset: str
    cache_type: str = Field(default="all", description="缓存类型：database | query | encrypted | all")
    message: str
    build_time_ms: float = 0