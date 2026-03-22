"""
CNN 图像检索 API 端点

提供基于 SkNN 的隐私保护图像检索接口
"""

import torch
from fastapi import APIRouter, File, UploadFile, HTTPException, Query
from torchvision import transforms
from PIL import Image
import io
import logging
from typing import Optional, Dict, Any

from cirtorch.services.sknn_service import SknnService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/cir", tags=["CNN 图像检索"])

# 全局服务实例（懒加载）
_sknn_service: Optional[SknnService] = None


def get_sknn_service() -> SknnService:
    """获取或创建 SknnService 单例。"""
    global _sknn_service
    if _sknn_service is None:
        _sknn_service = SknnService()
    return _sknn_service


@router.post("/privacy_search")
async def privacy_search(
    image: UploadFile = File(...),
    top_k: int = Query(default=10, ge=1, le=100)
) -> Dict[str, Any]:
    """
    隐私保护图像检索（密文搜索）。

    使用 SkNN 方案对查询图像进行加密，然后在加密数据库中搜索相似图像。
    整个过程服务器无法获取明文特征，保护用户隐私。
    """
    service = get_sknn_service()

    # 检查数据库是否已加载
    if service.db_features is None:
        raise HTTPException(400, "检索数据库未加载，请先构建或加载数据库")

    if not image.content_type.startswith("image/"):
        raise HTTPException(400, "无效的图像格式")

    try:
        # 读取图像
        image_bytes = await image.read()
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')

        # 执行检索
        results = service.search_by_image(img, top_k=top_k)

        return {
            "status": "success",
            "query_processed": True,
            "results": results
        }

    except Exception as e:
        logger.error(f"检索失败：{str(e)}")
        raise HTTPException(500, f"检索过程中出错：{str(e)}")


@router.post("/search")
async def search(
    image: UploadFile = File(...),
    top_k: int = Query(default=10, ge=1, le=100),
    use_encrypted: bool = Query(default=True, description="是否使用加密检索")
) -> Dict[str, Any]:
    """
    图像检索（支持明文/密文两种模式）。

    参数：
        image: 查询图像
        top_k: 返回结果数量
        use_encrypted: 是否使用加密检索（默认 True）
    """
    service = get_sknn_service()

    if service.db_features is None:
        raise HTTPException(400, "检索数据库未加载")

    if not image.content_type.startswith("image/"):
        raise HTTPException(400, "无效的图像格式")

    try:
        image_bytes = await image.read()
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')

        if use_encrypted:
            # 加密检索
            results = service.search_by_image(img, top_k=top_k)
        else:
            # 明文检索（用于调试对比）
            results = service.search_by_image(img, top_k=top_k, encrypted=False)

        return {
            "status": "success",
            "use_encrypted": use_encrypted,
            "results": results
        }

    except Exception as e:
        logger.error(f"检索失败：{str(e)}")
        raise HTTPException(500, f"检索过程中出错：{str(e)}")


@router.get("/status")
async def get_status() -> Dict[str, Any]:
    """获取检索服务状态。"""
    service = get_sknn_service()
    return service.get_status()


@router.post("/database/build")
async def build_database(
    image_dir: str,
    save_dir: Optional[str] = None
) -> Dict[str, Any]:
    """构建加密检索数据库。"""
    service = get_sknn_service()

    try:
        db_features, db_images = service.build_database(
            image_dir=image_dir,
            save_dir=save_dir
        )

        return {
            "status": "success",
            "database_size": len(db_images),
            "feature_shape": list(db_features.shape),
            "save_dir": save_dir or str(service.db_dir)
        }

    except Exception as e:
        logger.error(f"建库失败：{str(e)}")
        raise HTTPException(500, f"建库过程中出错：{str(e)}")


@router.post("/database/load")
async def load_database(
    db_dir: Optional[str] = None
) -> Dict[str, Any]:
    """加载已存在的数据库。"""
    service = get_sknn_service()

    try:
        service.load_database(db_dir)

        return {
            "status": "success",
            "database_size": len(service.db_image_names),
            "feature_shape": list(service.db_features.shape)
        }

    except Exception as e:
        logger.error(f"加载数据库失败：{str(e)}")
        raise HTTPException(500, f"加载数据库时出错：{str(e)}")


@router.post("/keys/generate")
async def generate_keys() -> Dict[str, Any]:
    """生成新的 SkNN 密钥。"""
    service = get_sknn_service()

    try:
        M1, M2, S = service.generate_keys()

        return {
            "status": "success",
            "feature_dim": service.feature_dim,
            "message": "新密钥已生成（请妥善保管）"
        }

    except Exception as e:
        logger.error(f"生成密钥失败：{str(e)}")
        raise HTTPException(500, f"密钥生成时出错：{str(e)}")
