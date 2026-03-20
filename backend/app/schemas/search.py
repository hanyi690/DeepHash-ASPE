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
    labels: Optional[List[List[int]]] = None


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


# ============== 搜索相关模式 ==============

class SearchRequest(BaseModel):
    """搜索请求。"""
    query_type: str = Field(..., description="查询类型：text 或 image")
    query_text: Optional[str] = None
    query_image: Optional[str] = None
    top_k: int = Field(default=10, ge=1, le=100)
    use_encrypted: bool = Field(default=True, description="是否使用加密检索")


class SearchResult(BaseModel):
    """单个搜索结果。"""
    rank: int
    image_id: int
    score: float
    distance: float
    captions: List[str] = []
    thumbnail_url: Optional[str] = None


class SearchResponse(BaseModel):
    """搜索响应。"""
    success: bool
    query_type: str
    query_text: Optional[str] = None
    results: List[SearchResult]
    total_results: int
    search_time_ms: float
    plaintext_map: Optional[float] = None
    ciphertext_map: Optional[float] = None
    map_difference: Optional[float] = None


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
    coco_status: Dict[str, Any]
