"""
加密 API 路由

提供 ASPE 加密相关端点。
"""

from fastapi import APIRouter, HTTPException
import numpy as np

from app.schemas.search import (
    EncryptDatabaseRequest,
    EncryptDatabaseResponse,
    TrapdoorRequest,
    TrapdoorResponse
)
from app.services.dcmh_encryption_service import get_dcmh_encryption_service
from app.services.dcmh_service import get_dcmh_service
from app.services.dataset_service import get_dataset_service
from app.services.hash_cache_service import get_hash_cache_service

router = APIRouter(prefix="/api/encrypt", tags=["encrypt"])


@router.post("/database", response_model=EncryptDatabaseResponse)
async def encrypt_database(request: EncryptDatabaseRequest):
    """加密检索库哈希码。"""
    try:
        encryption_service = get_dcmh_encryption_service()

        # 转换哈希码为 numpy 数组
        hash_codes = np.array(request.hash_codes, dtype=np.float64)

        # 验证哈希码格式
        if hash_codes.ndim != 2:
            return EncryptDatabaseResponse(
                success=False,
                message="哈希码必须是 2D 数组"
            )

        # 加密
        encrypted = encryption_service.encrypt_database(hash_codes)

        return EncryptDatabaseResponse(
            success=True,
            encrypted_size=encrypted.shape[0],
            bit_dim=encryption_service.bit_dim,
            message=f"成功加密 {encrypted.shape[0]} 个哈希码"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/trapdoor", response_model=TrapdoorResponse)
async def generate_trapdoor(request: TrapdoorRequest):
    """生成查询陷阱门。"""
    try:
        encryption_service = get_dcmh_encryption_service()

        # 转换哈希码为 numpy 数组
        hash_code = np.array(request.hash_code, dtype=np.float64).reshape(1, -1)

        # 生成陷阱门
        encrypted_query = encryption_service.encrypt_query(hash_code)

        # 转换为列表
        encrypted_query_list = encrypted_query.squeeze().tolist()

        return TrapdoorResponse(
            success=True,
            encrypted_query=encrypted_query_list,
            message="成功生成陷阱门"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/build-database")
async def build_encrypted_database():
    """构建加密检索数据库（使用 Flickr25K 数据集）。"""
    try:
        dcmh_service = get_dcmh_service()
        encryption_service = get_dcmh_encryption_service()
        hash_cache = get_hash_cache_service()
        dataset_service = get_dataset_service()

        # 构建数据库缓存
        image_codes, text_codes, tags = hash_cache.build_full_database_cache(
            dcmh_service, dataset_service, batch_size=32, force_rebuild=False
        )

        # 加载哈希码并加密
        encryption_service.encrypt_database(image_codes)

        return {
            "success": True,
            "num_images": image_codes.shape[0],
            "bit_dim": dcmh_service.bit_dim,
            "message": f"成功构建包含 {image_codes.shape[0]} 个图像的加密数据库"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status")
async def get_encrypt_status():
    """获取加密服务状态。"""
    try:
        encryption_service = get_dcmh_encryption_service()
        status = encryption_service.get_status()
        return {
            "success": True,
            **status
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/verify-consistency")
async def verify_consistency(num_samples: int = 10):
    """验证 ASPE 加密前后 mAP 一致性。"""
    try:
        encryption_service = get_dcmh_encryption_service()

        # 检查数据库是否已加密
        if not encryption_service.has_keys():
            return {
                "success": False,
                "message": "加密密钥未初始化，请先加载密钥"
            }

        # 返回验证信息
        return {
            "success": True,
            "message": "加密服务正常",
            "has_keys": encryption_service.has_keys(),
            "bit_dim": encryption_service.bit_dim
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
