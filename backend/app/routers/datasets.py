"""
数据集管理 API 路由

提供数据集列表、状态查询和缓存管理功能。
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Dict, Any

from app.schemas.dataset import (
    DatasetInfo,
    DatasetListResponse,
    DatasetStatus,
    DatasetStatusResponse,
    SystemStatusResponse,
    CacheBuildRequest,
    CacheBuildResponse
)
from app.schemas.search import DCMH_DATASETS, CIR_DATASETS
from app.services.dataset_service import get_dataset_service, DATASET_CONFIGS
from app.services.hash_cache_service import get_hash_cache_service
from app.services.dcmh_service import get_dcmh_service

import torch

router = APIRouter(prefix="/api/datasets", tags=["datasets"])


@router.get("", response_model=DatasetListResponse)
async def list_datasets():
    """
    获取所有支持的数据集列表。

    返回 DCMH 和 CIR 数据集的基本信息。
    """
    hash_cache = get_hash_cache_service()
    datasets = []

    # DCMH 数据集
    for name in DCMH_DATASETS:
        config = DATASET_CONFIGS.get(name, {})
        cache_exists = hash_cache.dcmh_cache_exists(name, "database")

        datasets.append(DatasetInfo(
            name=name,
            display_name=config.get('display_name', name.upper()),
            type='dcmh',
            status='ready' if cache_exists else 'not_cached',
            cache_exists=cache_exists,
            image_count=config.get('database_size', 0)
        ))

    # CIR 数据集
    for name in CIR_DATASETS:
        cache_exists = hash_cache.cir_cache_exists(name, "features")

        datasets.append(DatasetInfo(
            name=name,
            display_name=name.upper(),
            type='cir',
            status='ready' if cache_exists else 'not_cached',
            cache_exists=cache_exists,
            image_count=0  # CIR 数据集大小需要从缓存获取
        ))

    return DatasetListResponse(
        success=True,
        datasets=datasets,
        message=f"找到 {len(datasets)} 个数据集"
    )


@router.get("/system/status")
async def get_system_status():
    """
    获取系统整体状态。

    返回 GPU 信息和所有数据集的状态。
    """
    hash_cache = get_hash_cache_service()

    # GPU 信息
    gpu_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if gpu_available else None
    device = "cuda" if gpu_available else "cpu"

    # 各数据集状态
    datasets_status = {}

    # DCMH 数据集
    for name in DCMH_DATASETS:
        dcmh_service = get_dcmh_service(dataset=name)
        # 优先检测新格式缓存 (database_image.npz)，兼容旧格式 (database.npz)
        database_codes, _ = hash_cache.load_dcmh_cache(name, "database_image")
        if database_codes is None:
            database_codes, _ = hash_cache.load_dcmh_cache(name, "database")

        datasets_status[name] = DatasetStatus(
            name=name,
            cache_loaded=database_codes is not None,
            cache_path=str(hash_cache.get_dcmh_cache_path(name, "database_image")),
            model_loaded=dcmh_service.is_loaded(),
            gpu_available=gpu_available,
            gpu_name=gpu_name,
            database_size=database_codes.shape[0] if database_codes is not None else 0,
            query_size=dcmh_service.y_dim,  # 使用 y_dim 作为查询大小的代理
            bit_dim=dcmh_service.bit_dim
        )

    # CIR 数据集
    for name in CIR_DATASETS:
        features, _ = hash_cache.load_cir_features(name)

        datasets_status[name] = DatasetStatus(
            name=name,
            cache_loaded=features is not None,
            cache_path=str(hash_cache.get_cir_cache_path(name, "features")),
            model_loaded=False,
            gpu_available=gpu_available,
            gpu_name=gpu_name,
            database_size=features.shape[0] if features is not None else 0,
            query_size=0,
            bit_dim=0
        )

    return SystemStatusResponse(
        success=True,
        gpu_available=gpu_available,
        gpu_name=gpu_name,
        device=device,
        datasets=datasets_status
    )


@router.get("/{name}/status", response_model=DatasetStatusResponse)
async def get_dataset_status(name: str):
    """
    获取指定数据集的详细状态。

    参数：
        name: 数据集名称（flickr25k, nuswide, roxford5k, rparis6k）
    """
    hash_cache = get_hash_cache_service()

    # 检查数据集是否支持
    if name in DCMH_DATASETS:
        # DCMH 数据集
        dataset_service = get_dataset_service(dataset_name=name)
        dcmh_service = get_dcmh_service(dataset=name)

        # 获取缓存信息 - 优先检测新格式，兼容旧格式
        database_codes, _ = hash_cache.load_dcmh_cache(name, "database_image")
        if database_codes is None:
            database_codes, _ = hash_cache.load_dcmh_cache(name, "database")

        status = DatasetStatus(
            name=name,
            cache_loaded=database_codes is not None,
            cache_path=str(hash_cache.get_dcmh_cache_path(name, "database_image")),
            model_loaded=dcmh_service.is_loaded(),
            gpu_available=torch.cuda.is_available(),
            gpu_name=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            database_size=database_codes.shape[0] if database_codes is not None else 0,
            query_size=dataset_service.query_size,
            bit_dim=dcmh_service.bit_dim
        )

    elif name in CIR_DATASETS:
        # CIR 数据集
        features, _ = hash_cache.load_cir_features(name)

        status = DatasetStatus(
            name=name,
            cache_loaded=features is not None,
            cache_path=str(hash_cache.get_cir_cache_path(name, "features")),
            model_loaded=False,  # CIR 模型状态需要从 CIR 服务获取
            gpu_available=torch.cuda.is_available(),
            gpu_name=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            database_size=features.shape[0] if features is not None else 0,
            query_size=0,
            bit_dim=0
        )

    else:
        raise HTTPException(
            status_code=404,
            detail=f"不支持的数据集：{name}，支持的数据集：{DCMH_DATASETS + CIR_DATASETS}"
        )

    return DatasetStatusResponse(
        success=True,
        status=status,
        message=f"数据集 {name} 状态获取成功"
    )


@router.post("/{name}/cache/build", response_model=CacheBuildResponse)
async def build_dataset_cache(name: str, request: CacheBuildRequest):
    """
    构建指定数据集的缓存。

    参数：
        name: 数据集名称
        request: 缓存构建请求参数
    """
    import time
    start_time = time.time()

    # 验证数据集
    if name not in DCMH_DATASETS and name not in CIR_DATASETS:
        raise HTTPException(
            status_code=404,
            detail=f"不支持的数据集：{name}"
        )

    try:
        hash_cache = get_hash_cache_service()

        if name in DCMH_DATASETS:
            # 构建 DCMH 缓存
            from app.services.dcmh_service import get_dcmh_service
            from app.services.dataset_service import get_dataset_service
            from app.services.aspe_service import get_aspe_service

            dcmh_service = get_dcmh_service(dataset=name)
            dataset_service = get_dataset_service(dataset_name=name)
            aspe_service = get_aspe_service()

            if not dcmh_service.is_loaded():
                raise HTTPException(
                    status_code=400,
                    detail="DCMH 模型未加载，无法构建缓存"
                )

            # 构建数据库缓存
            database_codes, database_labels = hash_cache.build_database_cache(
                dcmh_service, dataset_service,
                batch_size=32, force_rebuild=request.force_rebuild, dataset=name
            )

            # 构建加密缓存
            hash_cache.build_encrypted_cache(
                aspe_service, force_rebuild=request.force_rebuild, dataset=name
            )

            build_time_ms = (time.time() - start_time) * 1000

            return CacheBuildResponse(
                success=True,
                dataset=name,
                cache_type="all",
                message=f"DCMH 缓存构建完成 ({name})",
                build_time_ms=build_time_ms
            )

        else:
            # CIR 缓存构建需要指定数据目录
            raise HTTPException(
                status_code=400,
                detail="CIR 数据集缓存构建需要使用 /api/cir/cache/build 端点"
            )

    except HTTPException:
        raise
    except Exception as e:
        build_time_ms = (time.time() - start_time) * 1000
        return CacheBuildResponse(
            success=False,
            dataset=name,
            cache_type="all",
            message=f"缓存构建失败：{str(e)}",
            build_time_ms=build_time_ms
        )


@router.delete("/{name}/cache")
async def clear_dataset_cache(name: str):
    """
    清除指定数据集的缓存。

    参数：
        name: 数据集名称
    """
    import os
    import shutil

    hash_cache = get_hash_cache_service()

    if name in DCMH_DATASETS:
        cache_path = hash_cache.get_dcmh_cache_path(name, "database").parent
        if cache_path.exists():
            shutil.rmtree(cache_path)
            return {"success": True, "message": f"已清除 {name} 的 DCMH 缓存"}
        else:
            return {"success": True, "message": f"{name} 缓存不存在"}

    elif name in CIR_DATASETS:
        cache_path = hash_cache.get_cir_cache_path(name, "features").parent
        if cache_path.exists():
            shutil.rmtree(cache_path)
            return {"success": True, "message": f"已清除 {name} 的 CIR 缓存"}
        else:
            return {"success": True, "message": f"{name} 缓存不存在"}

    else:
        raise HTTPException(
            status_code=404,
            detail=f"不支持的数据集：{name}"
        )