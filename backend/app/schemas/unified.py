"""
统一数据类型定义

提供 DCMH 跨模态检索和 CIR 图像检索的统一类型定义。
支持三种检索模式和两种加密模式。
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from enum import Enum


# ============== 枚举类型 ==============

class SearchMode(str, Enum):
    """检索模式枚举。"""
    TAG_TO_IMAGE = "tag_to_image"    # 标签搜图（DCMH）
    IMAGE_TO_TAG = "image_to_tag"    # 图搜标签（DCMH）
    IMAGE_TO_IMAGE = "image_to_image"  # 图搜图（CIR）


class EncryptionMode(str, Enum):
    """加密模式枚举。"""
    PLAINTEXT = "plaintext"  # 明文检索
    ENCRYPTED = "encrypted"  # 加密检索


# ============== 数据集常量 ==============

DCMH_DATASETS = ['flickr25k', 'nuswide']
CIR_DATASETS = ['roxford5k', 'rparis6k']
ALL_DATASETS = DCMH_DATASETS + CIR_DATASETS


# ============== 统一结果类型 ==============

class BaseSearchResult(BaseModel):
    """
    基础检索结果。

    统一 DCMH 和 CIR 的检索结果格式。
    """
    rank: int = Field(..., description="结果排名")
    image_id: str = Field(..., description="图像标识符（统一为字符串）")
    score: float = Field(..., description="相似度分数（越高越相似）")
    distance: float = Field(..., description="距离（越小越相似）")
    thumbnail_url: Optional[str] = Field(None, description="缩略图 URL")


class TagToImageResult(BaseSearchResult):
    """
    标签→图像检索结果。

    继承基础结果，增加标签和类别信息。
    """
    tags: List[int] = Field(default=[], description="图像标签索引列表")
    tag_names: List[str] = Field(default=[], description="标签名称列表")
    hit_tags: List[int] = Field(default=[], description="命中的查询标签索引")
    hit_tag_names: List[str] = Field(default=[], description="命中标签名称")
    category_hit: bool = Field(default=False, description="LAll 类别是否命中")
    tag_hit: bool = Field(default=False, description="YAll 标签是否命中")
    category_names: List[str] = Field(default=[], description="结果图像类别名称")
    hit_category_names: List[str] = Field(default=[], description="命中类别名称")
    hash_code: Optional[List[int]] = Field(None, description="哈希码展示")


class ImageToTagResult(BaseModel):
    """
    图像→标签检索结果。

    返回相似图像的标签信息。
    """
    rank: int = Field(..., description="结果排名")
    image_id: str = Field(..., description="来源图像标识符")
    tags: List[int] = Field(default=[], description="标签索引列表")
    tag_names: List[str] = Field(default=[], description="标签名称列表")
    score: float = Field(..., description="相似度分数")
    distance: float = Field(..., description="距离")
    thumbnail_url: Optional[str] = Field(None, description="来源图像缩略图 URL")
    category_names: List[str] = Field(default=[], description="结果图像类别名称")


class ImageToImageResult(BaseSearchResult):
    """
    图像→图像检索结果（CIR）。

    继承基础结果，增加图像名称信息。
    """
    image_name: str = Field(default="", description="图像文件名")
    image_url: Optional[str] = Field(None, description="图像访问 URL")


# ============== 统一请求类型 ==============

class UnifiedSearchRequest(BaseModel):
    """
    统一检索请求。

    支持三种检索模式：
    - tag_to_image: 标签搜图，需要 tag_indices
    - image_to_tag: 图搜标签，需要 query_image (Base64)
    - image_to_image: 图搜图，需要 query_image (Base64)
    """
    mode: SearchMode = Field(
        default=SearchMode.TAG_TO_IMAGE,
        description="检索模式：tag_to_image | image_to_tag | image_to_image"
    )
    encryption: EncryptionMode = Field(
        default=EncryptionMode.ENCRYPTED,
        description="加密模式：plaintext | encrypted"
    )
    dataset: str = Field(
        default="flickr25k",
        description="数据集名称"
    )
    top_k: int = Field(
        default=10,
        ge=1,
        le=100,
        description="返回结果数量"
    )
    tag_indices: Optional[List[int]] = Field(
        default=None,
        description="标签索引列表（tag_to_image 模式必需）"
    )
    query_image: Optional[str] = Field(
        default=None,
        description="查询图像 Base64 编码（image_to_tag 和 image_to_image 模式必需）"
    )


class UnifiedSearchUploadRequest(BaseModel):
    """
    文件上传检索请求。

    用于处理 multipart/form-data 文件上传。
    """
    mode: SearchMode = Field(
        default=SearchMode.IMAGE_TO_IMAGE,
        description="检索模式"
    )
    encryption: EncryptionMode = Field(
        default=EncryptionMode.ENCRYPTED,
        description="加密模式"
    )
    dataset: str = Field(
        default="roxford5k",
        description="数据集名称"
    )
    top_k: int = Field(
        default=10,
        ge=1,
        le=100,
        description="返回结果数量"
    )


# ============== 统一响应类型 ==============

class HitStats(BaseModel):
    """命中率统计。"""
    total_results: int = Field(default=0, description="结果总数")
    # 标签命中（YAll）
    tag_hits: int = Field(default=0, description="标签命中数")
    tag_hit_rate: float = Field(default=0.0, description="标签命中率")
    # 类别命中（LAll）
    category_hits: int = Field(default=0, description="类别命中数")
    category_hit_rate: float = Field(default=0.0, description="类别命中率")
    # 查询信息
    query_tags: List[int] = Field(default=[], description="查询标签索引")
    query_tag_names: List[str] = Field(default=[], description="查询标签名称")
    # 类别分布（image_to_tag 模式）
    category_distribution: Optional[Dict[str, int]] = Field(
        None, description="类别分布统计"
    )


class EncryptionInfo(BaseModel):
    """加密状态信息。"""
    method: str = Field(default="ASPE", description="加密方法")
    query_encrypted: bool = Field(default=False, description="查询是否加密")
    database_encrypted: bool = Field(default=False, description="数据库是否加密")
    security_level: int = Field(default=3, description="安全等级")
    bit_dim: Optional[int] = Field(None, description="哈希码位数（DCMH）")
    feature_dim: Optional[int] = Field(None, description="特征维度（CIR）")


class UnifiedSearchResponse(BaseModel):
    """
    统一检索响应。

    支持三种检索模式的结果返回。
    """
    success: bool = Field(..., description="请求是否成功")
    mode: SearchMode = Field(..., description="检索模式")
    encryption: EncryptionMode = Field(..., description="加密模式")
    dataset: str = Field(..., description="数据集名称")

    # 结果列表
    results: List[BaseSearchResult] = Field(
        default=[],
        description="检索结果列表（tag_to_image 和 image_to_image）"
    )
    tag_results: List[ImageToTagResult] = Field(
        default=[],
        description="标签结果列表（image_to_tag）"
    )

    # 统计信息
    total_results: int = Field(default=0, description="结果总数")
    search_time_ms: float = Field(default=0.0, description="检索耗时（毫秒）")
    hit_stats: Optional[HitStats] = Field(None, description="命中率统计")

    # 查询信息
    query_tag_names: List[str] = Field(default=[], description="查询标签名称")
    query_hash_code: Optional[List[int]] = Field(None, description="查询哈希码")

    # 加密信息
    encryption_info: Optional[EncryptionInfo] = Field(None, description="加密信息")

    # mAP 信息
    plaintext_map: Optional[float] = Field(None, description="明文 mAP")
    ciphertext_map: Optional[float] = Field(None, description="密文 mAP")
    map_difference: Optional[float] = Field(None, description="mAP 差异")

    # 错误信息
    message: Optional[str] = Field(None, description="错误或提示信息")


# ============== 服务状态类型 ==============

class ServiceStatus(BaseModel):
    """服务状态响应。"""
    success: bool = Field(default=True, description="请求是否成功")
    service_type: str = Field(..., description="服务类型：dcmh | cir")
    dataset: str = Field(..., description="数据集名称")
    initialized: bool = Field(default=False, description="是否已初始化")
    model_loaded: bool = Field(default=False, description="模型是否已加载")
    plaintext_indexed: bool = Field(default=False, description="明文索引是否已构建")
    encrypted_indexed: bool = Field(default=False, description="加密索引是否已构建")
    index_size: int = Field(default=0, description="索引大小")
    keys_loaded: bool = Field(default=False, description="密钥是否已加载")
    additional_info: Optional[Dict[str, Any]] = Field(None, description="附加信息")


class UnifiedStatusResponse(BaseModel):
    """统一状态响应。"""
    success: bool = Field(default=True)
    dcmh_status: Optional[ServiceStatus] = None
    cir_status: Optional[ServiceStatus] = None
    key_manager_status: Optional[Dict[str, Any]] = None