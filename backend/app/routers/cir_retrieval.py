"""
CNN Image Retrieval API Routes

支持两种检索模式：
1. 明文检索（传统 CNN 特征相似度）
2. 隐私检索（ASPE 加密保护）
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

from app.services.cir_service import get_cir_service, CIRService
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
    2. 使用 ASPE 方案加密特征
    3. 保存加密数据库和密钥
    """
    service = get_cir_service()

    try:
        db_features, db_images = service.build_database(
            image_dir=image_dir,
            save_dir=save_dir
        )

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
    service = get_cir_service()

    try:
        service.load_database(db_dir)

        return {
            "status": "success",
            "database_size": service.get_index_size(),
            "feature_shape": list(service.db_features.shape) if service.db_features else None
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
    service = get_cir_service()

    if not service.is_indexed:
        raise HTTPException(status_code=400, detail="加密数据库未加载")

    try:
        results = service.search(
            query_image_path=request.query_image_path,
            top_k=request.top_k
        )

        return {
            "status": "success",
            "use_encrypted": True,
            "results": [r.__dict__ for r in results]
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
    service = get_cir_service()

    if not service.is_indexed:
        raise HTTPException(status_code=400, detail="加密数据库未加载")

    try:
        # 保存上传的图像到临时文件
        import tempfile
        image_bytes = await image.read()

        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            tmp.write(image_bytes)
            tmp_path = tmp.name

        # 执行检索
        results = service.search(query_image_path=tmp_path, top_k=top_k)

        # 清理临时文件
        import os
        os.unlink(tmp_path)

        return {
            "status": "success",
            "use_encrypted": True,
            "results": [r.__dict__ for r in results]
        }

    except Exception as e:
        logger.error(f"检索失败：{str(e)}")
        raise HTTPException(status_code=500, detail=f"检索失败：{str(e)}")


@router.post("/search/upload")
async def cir_search_upload(
    image: UploadFile = File(..., description="查询图像"),
    top_k: int = Query(default=10, ge=1, le=100)
) -> Dict[str, Any]:
    """
    明文 CNN 图像检索（文件上传模式）。

    不使用加密，直接计算特征相似度进行检索。
    适用于对隐私保护没有要求的场景。
    """
    service = get_cir_service()

    if not service.is_indexed:
        raise HTTPException(status_code=400, detail="数据库未加载，请先加载或构建数据库")

    try:
        # 保存上传的图像到临时文件
        import tempfile
        image_bytes = await image.read()

        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            tmp.write(image_bytes)
            tmp_path = tmp.name

        # 执行明文检索
        results = service.search(query_image_path=tmp_path, top_k=top_k)

        # 清理临时文件
        import os
        os.unlink(tmp_path)

        return {
            "status": "success",
            "use_encrypted": False,
            "results": [r.__dict__ for r in results]
        }

    except Exception as e:
        logger.error(f"明文检索失败：{str(e)}")
        raise HTTPException(status_code=500, detail=f"检索失败：{str(e)}")


@router.get("/sknn/status")
async def get_sknn_status() -> Dict[str, Any]:
    """获取 SkNN 服务状态。"""
    service = get_cir_service()
    return service.get_status()


@router.post("/sknn/verify")
async def verify_encryption() -> Dict[str, Any]:
    """验证 SkNN 加密的内积保持性。"""
    aspe = ASPEForCNN()
    result = aspe.verify_inner_product_preservation(num_samples=10)

    return {
        "status": "success",
        "verification": result
    }


@router.get("/sknn/database/info")
async def get_database_info() -> Dict[str, Any]:
    """
    获取加密数据库详细信息。

    返回数据库中的图像列表和元数据。
    """
    service = get_cir_service()
    status = service.get_status()

    if not status.get("indexed"):
        return {
            "loaded": False,
            "message": "数据库未加载，请先加载或构建数据库"
        }

    return {
        "loaded": True,
        "database_size": status.get("index_size", 0),
        "feature_shape": [2 * service.feature_dim, status.get("index_size", 0)] if service.db_features else None,
        "images": service.db_image_names[:20] if service.db_image_names else [],
        "total_images": len(service.db_image_names) if service.db_image_names else 0
    }


@router.post("/sknn/database/load-demo")
async def load_demo_database() -> Dict[str, Any]:
    """
    加载演示数据库。

    如果演示数据库不存在，返回提示信息。
    """
    demo_db_path = Path('data/cir_demo_db')

    if not demo_db_path.exists():
        return {
            "status": "error",
            "message": "演示数据库不存在，请先运行构建脚本",
            "hint": "运行: python scripts/build_cir_demo_db.py --image-dir <图像目录> --save-dir data/cir_demo_db"
        }

    return await load_sknn_database(str(demo_db_path))
