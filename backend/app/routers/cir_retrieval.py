"""
CNN Image Retrieval API Routes

支持两种检索模式：
1. 明文检索（传统 CNN 特征相似度）
2. 隐私检索（SkNN 加密保护）
"""

from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, Query, Form
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from pathlib import Path
import sys
import io
import logging

from PIL import Image

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.services.cir_service import get_cir_service
from core.aspe.cnn_wrapper import ASPEForCNN

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/cir", tags=["CNN Image Retrieval"])


# ============= 请求/响应模型 =============

class BuildIndexRequest(BaseModel):
    """Request model for building index."""
    image_paths: List[str] = Field(..., description="List of image paths to index")
    image_ids: Optional[List[str]] = Field(None, description="Optional list of image IDs")


class SearchRequest(BaseModel):
    """Request model for image search."""
    query_image_path: str = Field(..., description="Path to query image")
    top_k: int = Field(default=10, ge=1, le=100, description="Number of results to return")
    use_encrypted: bool = Field(default=True, description="Whether to use encrypted search")


class PrivacySearchRequest(BaseModel):
    """Request model for privacy-preserving search."""
    image_file: UploadFile = File(..., description="Query image file")
    top_k: int = Field(default=10, ge=1, le=100)


class LoadIndexRequest(BaseModel):
    """Request model for loading index."""
    index_dir: str = Field(..., description="Directory containing index files")


class SaveIndexRequest(BaseModel):
    """Request model for saving index."""
    save_dir: str = Field(..., description="Directory to save index files")


class SknnBuildDatabaseRequest(BaseModel):
    """Request model for building SkNN encrypted database."""
    image_dir: str = Field(..., description="Directory of images to index")
    save_dir: str = Field(..., description="Directory to save encrypted database")
    feature_dim: int = Field(default=2048, description="Feature dimension")


class SknnSearchRequest(BaseModel):
    """Request model for SkNN privacy search."""
    query_image_path: str = Field(..., description="Path to query image")
    top_k: int = Field(default=10, ge=1, le=100)


# ============= 全局状态 =============

# SkNN 服务实例（懒加载）
_sknn_service: Optional[ASPEForCNN] = None
_sknn_encrypted_db: Optional[Any] = None


def get_sknn_service() -> ASPEForCNN:
    """Get or create SkNN service instance."""
    global _sknn_service
    if _sknn_service is None:
        _sknn_service = ASPEForCNN()
    return _sknn_service


# ============= CNN 检索端点（明文模式） =============

@router.get("/status")
async def get_status() -> Dict[str, Any]:
    """Get CIR service status."""
    service = get_cir_service()
    return service.get_status()


@router.get("/initialize")
async def initialize_service(
    architecture: str = "resnet50",
    pooling: str = "gem",
    whitening: bool = True
) -> Dict[str, Any]:
    """Initialize the CIR service with specified parameters."""
    service = get_cir_service()
    service.initialize(
        architecture=architecture,
        pooling=pooling,
        whitening=whitening
    )
    return {
        "status": "success",
        "message": f"CIR service initialized with {architecture}-{pooling}"
    }


@router.post("/index/build")
async def build_index(request: BuildIndexRequest) -> Dict[str, Any]:
    """Build feature index from a collection of images."""
    service = get_cir_service()
    result = service.build_index(
        image_paths=request.image_paths,
        image_ids=request.image_ids
    )
    if result.get("status") == "error":
        raise HTTPException(status_code=500, detail=result.get("message"))
    return result


@router.post("/index/save")
async def save_index(request: SaveIndexRequest) -> Dict[str, Any]:
    """Save the current index to disk."""
    service = get_cir_service()
    result = service.save_index(request.save_dir)
    if result.get("status") == "error":
        raise HTTPException(status_code=500, detail=result.get("message"))
    return result


@router.post("/index/load")
async def load_index(request: LoadIndexRequest) -> Dict[str, Any]:
    """Load index from disk."""
    service = get_cir_service()
    result = service.load_index(request.index_dir)
    if result.get("status") == "error":
        raise HTTPException(status_code=500, detail=result.get("message"))
    return result


@router.post("/search")
async def search_images(request: SearchRequest) -> Dict[str, Any]:
    """
    Search for similar images.

    支持明文和加密两种检索模式。
    """
    service = get_cir_service()
    result = service.search(
        query_image_path=request.query_image_path,
        top_k=request.top_k
    )
    if result.get("status") == "error":
        raise HTTPException(status_code=500, detail=result.get("message"))
    return result


@router.get("/search/{query_image_path:path}")
async def search_images_get(
    query_image_path: str,
    top_k: int = 10
) -> Dict[str, Any]:
    """Search for similar images (GET endpoint)."""
    service = get_cir_service()
    result = service.search(
        query_image_path=query_image_path,
        top_k=top_k
    )
    if result.get("status") == "error":
        raise HTTPException(status_code=500, detail=result.get("message"))
    return result


@router.get("/feature/{image_path:path}")
async def extract_feature(image_path: str) -> Dict[str, Any]:
    """Extract feature vector from an image."""
    service = get_cir_service()
    result = service.extract_feature(image_path)
    if result.get("status") == "error":
        raise HTTPException(status_code=500, detail=result.get("message"))
    return result


# ============= SkNN 隐私检索端点 =============

@router.post("/sknn/keys/generate")
async def generate_sknn_keys(
    feature_dim: int = Query(default=2048, description="Feature dimension"),
    seed: Optional[int] = Query(default=None, description="Random seed")
) -> Dict[str, Any]:
    """
    生成新的 SkNN 密钥。

    密钥用于加密数据库和查询特征，必须妥善保管。
    """
    global _sknn_service
    _sknn_service = ASPEForCNN(feature_dim=feature_dim, seed=seed)

    return {
        "status": "success",
        "feature_dim": feature_dim,
        "message": "新 SkNN 密钥已生成（请妥善保管）"
    }


@router.post("/sknn/database/build")
async def build_sknn_database(
    image_dir: str = Query(..., description="图像目录"),
    save_dir: str = Query(..., description="数据库保存目录"),
    feature_dim: int = Query(default=2048, description="特征维度")
) -> Dict[str, Any]:
    """
    构建 SkNN 加密检索数据库。

    流程：
    1. 提取每张图像的 CNN 特征
    2. 使用 SkNN 方案加密特征
    3. 保存加密数据库和密钥
    """
    from cirtorch.services.sknn_service import SknnService

    try:
        service = SknnService(feature_dim=feature_dim)
        db_features, db_images = service.build_database(
            image_dir=image_dir,
            save_dir=save_dir
        )

        # 缓存到内存
        global _sknn_encrypted_db
        _sknn_encrypted_db = {
            'features': db_features,
            'image_names': db_images,
            'service': service
        }

        return {
            "status": "success",
            "database_size": len(db_images),
            "feature_shape": list(db_features.shape),
            "save_dir": save_dir
        }

    except Exception as e:
        logger.error(f"建库失败：{str(e)}")
        raise HTTPException(status_code=500, detail=f"建库失败：{str(e)}")


@router.post("/sknn/database/load")
async def load_sknn_database(
    db_dir: str = Query(..., description="数据库目录")
) -> Dict[str, Any]:
    """加载已存在的 SkNN 加密数据库。"""
    from cirtorch.services.sknn_service import SknnService

    try:
        service = SknnService(feature_dim=2048)
        service.load_database(db_dir)

        global _sknn_encrypted_db
        _sknn_encrypted_db = {
            'features': service.db_features,
            'image_names': service.db_image_names,
            'service': service
        }

        return {
            "status": "success",
            "database_size": len(service.db_image_names),
            "feature_shape": list(service.db_features.shape)
        }

    except Exception as e:
        logger.error(f"加载数据库失败：{str(e)}")
        raise HTTPException(status_code=500, detail=f"加载数据库失败：{str(e)}")


@router.post("/sknn/search")
async def sknn_search(
    request: SknnSearchRequest
) -> Dict[str, Any]:
    """
    SkNN 隐私保护图像检索。

    使用加密查询在加密数据库中搜索，服务器无法获取明文特征。
    """
    if _sknn_encrypted_db is None:
        raise HTTPException(status_code=400, detail="加密数据库未加载")

    service = _sknn_encrypted_db['service']

    try:
        results = service.search(
            query_image_path=request.query_image_path,
            top_k=request.top_k
        )

        return {
            "status": "success",
            "use_encrypted": True,
            "results": results
        }

    except Exception as e:
        logger.error(f"检索失败：{str(e)}")
        raise HTTPException(status_code=500, detail=f"检索失败：{str(e)}")


@router.post("/sknn/search/upload")
async def sknn_search_upload(
    image: UploadFile = File(..., description="查询图像"),
    top_k: int = Query(default=10, ge=1, le=100)
) -> Dict[str, Any]:
    """
    SkNN 隐私保护图像检索（文件上传模式）。

    用户上传图像，服务器在加密数据库中搜索相似图像。
    """
    if _sknn_encrypted_db is None:
        raise HTTPException(status_code=400, detail="加密数据库未加载")

    service = _sknn_encrypted_db['service']

    try:
        # 读取图像
        image_bytes = await image.read()
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')

        # 执行检索
        results = service.search_by_image(img, top_k=top_k)

        return {
            "status": "success",
            "use_encrypted": True,
            "results": results
        }

    except Exception as e:
        logger.error(f"检索失败：{str(e)}")
        raise HTTPException(status_code=500, detail=f"检索失败：{str(e)}")


@router.get("/sknn/status")
async def get_sknn_status() -> Dict[str, Any]:
    """获取 SkNN 服务状态。"""
    global _sknn_service, _sknn_encrypted_db

    status = {
        "keys_generated": _sknn_service is not None and _sknn_service.M1 is not None,
        "database_loaded": _sknn_encrypted_db is not None,
        "database_size": len(_sknn_encrypted_db['image_names']) if _sknn_encrypted_db else 0
    }

    if _sknn_service:
        status.update(_sknn_service.get_status())

    return status


@router.post("/sknn/verify")
async def verify_encryption() -> Dict[str, Any]:
    """验证 SkNN 加密的内积保持性。"""
    if _sknn_service is None:
        _sknn_service = ASPEForCNN()

    result = _sknn_service.verify_inner_product_preservation(num_samples=10)

    return {
        "status": "success",
        "verification": result
    }
