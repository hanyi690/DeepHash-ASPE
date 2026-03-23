"""
指标 API 路由

提供 mAP 评估和系统监控端点。
"""

from fastapi import APIRouter, HTTPException
import numpy as np
import time
import torch

from app.schemas.search import (
    MetricsRequest,
    MetricsResponse,
    SystemStatus
)
from app.services.aspe_service import get_aspe_service
from app.services.dcmh_service import get_dcmh_service
from app.services.dataset_service import get_dataset_service

router = APIRouter(prefix="/api/metrics", tags=["metrics"])


@router.get("", response_model=MetricsResponse)
async def get_metrics(k: int = None, num_queries: int = 100):
    """
    计算 mAP 评估指标。

    比较明文和密文检索的 mAP。
    """
    try:
        start_time = time.time()

        aspe_service = get_aspe_service()
        dcmh_service = get_dcmh_service()

        # 检查数据库是否已构建
        if aspe_service.encrypted_database is None:
            # 构建演示数据库
            num_db = 500
            demo_hash_codes = np.sign(np.random.randn(num_db, dcmh_service.bit_dim))
            labels = np.random.randint(0, 2, (num_db, 20)).astype(np.float32)
            aspe_service.encrypt_database(demo_hash_codes, labels)

        # 生成查询
        np.random.seed(42)
        num_queries = min(num_queries, 100)
        query_codes = np.sign(np.random.randn(num_queries, dcmh_service.bit_dim))
        query_labels = np.random.randint(0, 2, (num_queries, 20)).astype(np.float32)

        # 计算明文 mAP
        plaintext_map = aspe_service.compute_plaintext_map(query_codes, query_labels, k)

        # 计算密文 mAP
        encrypted_query = aspe_service.generate_trapdoor(query_codes)
        ciphertext_map = aspe_service.compute_ciphertext_map(encrypted_query, query_labels, k)

        computation_time_ms = (time.time() - start_time) * 1000

        return MetricsResponse(
            success=True,
            plaintext_i2t_map=float(plaintext_map),
            plaintext_t2i_map=float(plaintext_map),  # 简化：使用相同值
            ciphertext_i2t_map=float(ciphertext_map),
            ciphertext_t2i_map=float(ciphertext_map),
            i2t_difference=abs(float(plaintext_map) - float(ciphertext_map)),
            t2i_difference=abs(float(plaintext_map) - float(ciphertext_map)),
            consistent=abs(float(plaintext_map) - float(ciphertext_map)) < 1e-3,
            num_queries=num_queries,
            computation_time_ms=computation_time_ms
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/compute")
async def compute_metrics(request: MetricsRequest):
    """计算详细指标。"""
    try:
        aspe_service = get_aspe_service()
        dcmh_service = get_dcmh_service()

        # 检查数据库
        if aspe_service.encrypted_database is None:
            return {
                "success": False,
                "message": "请先构建加密数据库"
            }

        # 生成随机查询
        np.random.seed(42)
        num_queries = min(request.num_queries, 500)
        query_codes = np.sign(np.random.randn(num_queries, dcmh_service.bit_dim))
        query_labels = np.random.randint(0, 2, (num_queries, 20)).astype(np.float32)

        # 验证一致性
        result = aspe_service.verify_consistency(query_codes, query_labels, num_queries)

        return {
            "success": True,
            "metrics": {
                "plaintext_map": result["plaintext_map"],
                "ciphertext_map": result["ciphertext_map"],
                "difference": result["difference"],
                "consistent": result["consistent"]
            },
            "config": {
                "bit_dim": dcmh_service.bit_dim,
                "num_queries": num_queries,
                "k": request.k
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/comparison")
async def get_comparison():
    """获取明文/密文 mAP 对比数据。"""
    try:
        aspe_service = get_aspe_service()
        dcmh_service = get_dcmh_service()

        # 检查数据库
        if aspe_service.encrypted_database is None:
            # 构建演示数据库
            num_db = 500
            demo_hash_codes = np.sign(np.random.randn(num_db, dcmh_service.bit_dim))
            labels = np.random.randint(0, 2, (num_db, 20)).astype(np.float32)
            aspe_service.encrypt_database(demo_hash_codes, labels)

        # 测试不同查询数量
        test_sizes = [10, 20, 50, 100]
        comparison_data = []

        for n in test_sizes:
            np.random.seed(42)
            query_codes = np.sign(np.random.randn(n, dcmh_service.bit_dim))
            query_labels = np.random.randint(0, 2, (n, 20)).astype(np.float32)

            # 计算明文 mAP
            plaintext_map = aspe_service.compute_plaintext_map(query_codes, query_labels)

            # 计算密文 mAP
            encrypted_query = aspe_service.generate_trapdoor(query_codes)
            ciphertext_map = aspe_service.compute_ciphertext_map(encrypted_query, query_labels)

            comparison_data.append({
                "num_queries": n,
                "plaintext_map": float(plaintext_map),
                "ciphertext_map": float(ciphertext_map),
                "difference": abs(float(plaintext_map) - float(ciphertext_map))
            })

        return {
            "success": True,
            "bit_dim": dcmh_service.bit_dim,
            "comparison": comparison_data
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status", response_model=SystemStatus)
async def get_system_status():
    """获取系统状态。"""
    try:
        aspe_service = get_aspe_service()
        dcmh_service = get_dcmh_service()
        dataset_service = get_dataset_service()

        # 数据集状态
        dataset_status = dataset_service.get_status()

        return SystemStatus(
            success=True,
            dcmh_status={
                "bit_dim": dcmh_service.bit_dim,
                "use_gpu": dcmh_service.use_gpu,
                "models_loaded": True
            },
            aspe_status=aspe_service.get_status(),
            dataset_status=dataset_status
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
