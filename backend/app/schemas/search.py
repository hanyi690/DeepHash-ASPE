"""
搜索相关的 Pydantic 模式

定义 API 请求和响应的数据结构。
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


# ============== 图像相关模式 ==============

class ImageUploadResponse(BaseModel):
    """图像上传响应。"""
    success: bool
    image_id: Optional[int] = None
    image_url: Optional[str] = None
    message: str


class ImageFeatureRequest(BaseModel):
    """图像特征提取请求。"""
    image_id: Optional[int] = None
    image_url: Optional[str] = None
    image_data: Optional[str] = None  # base64 编码


class ImageFeatureResponse(BaseModel):
    """图像特征响应。"""
    success: bool
    features: Optional[List[float]] = None
    hash_code: Optional[List[int]] = None
    message: str


# ============== 文本相关模式 ==============

class TextProcessRequest(BaseModel):
    """文本处理请求。"""
    text: str = Field(..., min_length=1, description="输入文本")


class TextProcessResponse(BaseModel):
    """文本处理响应。"""
    success: bool
    text: str
    features: Optional[List[float]] = None
    hash_code: Optional[List[int]] = None
    message: str


# ============== 哈希码相关模式 ==============

class HashCodeRequest(BaseModel):
    """哈希码生成请求。"""
    input_type: str = Field(..., description="输入类型：image 或 text")
    input_data: Any = Field(..., description="输入数据")


class HashCodeResponse(BaseModel):
    """哈希码响应。"""
    success: bool
    hash_code: Optional[List[int]] = None
    bit_dim: int = 0
    message: str


# ============== 加密相关模式 ==============

class EncryptDatabaseRequest(BaseModel):
    """加密数据库请求。"""
    hash_codes: List[List[int]] = Field(..., description="哈希码列表")
    tags: Optional[List[List[int]]] = None


class EncryptDatabaseResponse(BaseModel):
    """加密数据库响应。"""
    success: bool
    encrypted_size: int = 0
    bit_dim: int = 0
    message: str


class TrapdoorRequest(BaseModel):
    """陷阱门生成请求。"""
    hash_code: List[int] = Field(..., description="查询哈希码")


class TrapdoorResponse(BaseModel):
    """陷阱门响应。"""
    success: bool
    encrypted_query: Optional[List[float]] = None
    message: str


# ============== 数据集相关常量 ==============

DCMH_DATASETS = ['flickr25k', 'nuswide']
CIR_DATASETS = ['roxford5k', 'rparis6k']


# ============== 搜索相关模式 ==============

class SearchRequest(BaseModel):
    """搜索请求。"""
    query_type: str = Field(
        default="tag_to_image",
        description="查询类型：tag_to_image | image_to_tag"
    )
    tag_indices: List[int] = Field(default=[], description="用户选择的标签索引")
    query_image: Optional[str] = None
    dataset: str = Field(default="flickr25k", description="数据集名称：flickr25k | nuswide")
    top_k: int = Field(default=10, ge=1, le=100)
    use_encrypted: bool = Field(default=True, description="是否使用加密检索")


class SearchResult(BaseModel):
    """单个搜索结果。"""
    rank: int
    image_id: int
    score: float
    distance: float
    tags: List[int] = []  # 该图像的标签索引列表 (YAll 索引)
    tag_names: List[str] = []  # 该图像的标签名称列表
    hit_tags: List[int] = []  # 命中的查询标签索引 (YAll 索引)
    hit_tag_names: List[str] = []  # 命中的查询标签名称
    thumbnail_url: Optional[str] = None
    hash_code: Optional[List[int]] = None  # 哈希码展示
    category_hit: bool = False  # LAll 类别是否命中
    tag_hit: bool = False  # YAll 标签是否命中
    category_names: List[str] = []  # 结果图像的所有类别名称 (LAll)
    hit_category_names: List[str] = []  # 命中的类别名称


class HitStats(BaseModel):
    """命中率统计。"""
    total_results: int
    # 标签命中（YAll）
    tag_hits: int
    tag_hit_rate: float
    # 类别命中（LAll）- 与评估 mAP 对应
    category_hits: int = 0
    category_hit_rate: float = 0.0
    # 查询信息
    query_tags: List[int] = []
    query_tag_names: List[str] = []


class ImageToTagResult(BaseModel):
    """图像→标签检索结果。"""
    rank: int
    image_id: int  # 来源图像ID
    tags: List[int]  # 标签索引 (YAll 索引)
    tag_names: List[str] = []  # 标签名称列表
    score: float
    distance: float
    thumbnail_url: Optional[str] = None  # 来源图像缩略图URL


class EncryptionInfo(BaseModel):
    """加密状态信息。"""
    method: str = "ASPE Scheme 1"
    query_encrypted: bool = False
    database_encrypted: bool = False
    security_level: int = 2
    bit_dim: int = 64


class SearchResponse(BaseModel):
    """搜索响应。"""
    success: bool
    query_type: str
    tag_indices: List[int] = []
    query_tag_names: List[str] = []  # 查询标签名称列表
    results: List[SearchResult] = []  # T2I/I2I 返回图像结果
    tag_results: List["ImageToTagResult"] = []  # I2T 返回标签结果
    total_results: int
    search_time_ms: float
    hit_stats: Optional[Dict[str, Any]] = None  # 命中率统计
    plaintext_map: Optional[float] = None
    ciphertext_map: Optional[float] = None
    map_difference: Optional[float] = None
    encryption_info: Optional[EncryptionInfo] = None  # 新增加密信息
    query_hash_code: Optional[List[int]] = None  # 查询哈希码


# ============== 指标相关模式 ==============

class MetricsRequest(BaseModel):
    """指标计算请求。"""
    k: Optional[int] = Field(default=None, description="截断位置")
    num_queries: int = Field(default=100, ge=1, le=1000)


class MetricsResponse(BaseModel):
    """指标响应。"""
    success: bool
    plaintext_i2t_map: float = 0.0
    plaintext_t2i_map: float = 0.0
    ciphertext_i2t_map: float = 0.0
    ciphertext_t2i_map: float = 0.0
    i2t_difference: float = 0.0
    t2i_difference: float = 0.0
    consistent: bool = True
    num_queries: int = 0
    computation_time_ms: float = 0.0


class SystemStatus(BaseModel):
    """系统状态响应。"""
    success: bool
    dcmh_status: Dict[str, Any]
    aspe_status: Dict[str, Any]
    dataset_status: Dict[str, Any]


# ============== 标签相关模式 ==============

class TagsResponse(BaseModel):
    """标签列表响应。"""
    success: bool
    total: int
    tag_dim: int  # 标签向量维度 (1386 for Flickr25K)
    message: str = "标签列表获取成功"
