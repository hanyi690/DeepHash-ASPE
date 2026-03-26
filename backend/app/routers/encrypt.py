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
from app.services.aspe_service import get_aspe_service
from app.services.dcmh_service import get_dcmh_service
from app.services.dataset_service import get_dataset_service
from app.services.hash_cache_service import get_hash_cache_service

router = APIRouter(prefix="/api/encrypt", tags=["encrypt"])


@router.post("/database", response_model=EncryptDatabaseResponse)
async def encrypt_database(request: EncryptDatabaseRequest):
    """加密检索库哈希码。"""
    try:
        aspe_service = get_aspe_service()

        # 转换哈希码为 numpy 数组
        hash_codes = np.array(request.hash_codes, dtype=np.float64)

        # 验证哈希码格式
        if hash_codes.ndim != 2:
            return EncryptDatabaseResponse(
                success=False,
                message="哈希码必须是 2D 数组"
            )

        # 处理标签
        labels = None
        if request.labels is not None:
            labels = np.array(request.labels, dtype=np.float32)

        # 加密
        encrypted = aspe_service.encrypt_database(hash_codes, labels)

        return EncryptDatabaseResponse(
            success=True,
            encrypted_size=encrypted.shape[0],
            bit_dim=encrypted.shape[1] - 1,  # ASPE 扩展 1 维
            message=f"成功加密 {encrypted.shape[0]} 个哈希码"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/trapdoor", response_model=TrapdoorResponse)
async def generate_trapdoor(request: TrapdoorRequest):
    """生成查询陷阱门。"""
    try:
        aspe_service = get_aspe_service()

        # 转换哈希码为 numpy 数组
        hash_code = np.array(request.hash_code, dtype=np.float64).reshape(1, -1)

        # 生成陷阱门
        encrypted_query = aspe_service.generate_trapdoor(hash_code)

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
        aspe_service = get_aspe_service()
        hash_cache = get_hash_cache_service()
        dataset_service = get_dataset_service()

        # 构建数据库缓存
        image_codes, text_codes, tags = hash_cache.build_full_database_cache(
            dcmh_service, dataset_service, batch_size=32, force_rebuild=False
        )

        # 构建加密缓存
        hash_cache.build_encrypted_cache(aspe_service, force_rebuild=False)

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
        aspe_service = get_aspe_service()
        status = aspe_service.get_status()
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
        aspe_service = get_aspe_service()

        # 检查数据库是否已加密
        if aspe_service.encrypted_database is None:
            return {
                "success": False,
                "message": "加密数据库未初始化，请先构建数据库"
            }

        # 生成随机查询用于测试
        np.random.seed(42)
        num_queries = min(num_samples, 100)
        query_codes = np.sign(np.random.randn(num_queries, aspe_service.bit_dim))
        query_labels = np.random.randint(0, 2, (num_queries, 10)).astype(np.float32)

        # 验证一致性
        result = aspe_service.verify_consistency(query_codes, query_labels, num_samples)

        return {
            "success": True,
            **result
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
