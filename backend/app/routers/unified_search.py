"""
统一检索 API 路由

提供统一的检索接口，支持：
- DCMH 跨模态检索（tag_to_image, image_to_tag）
- CIR 图像检索（image_to_image）
- 明文/加密检索

端点设计：
- POST /api/search - JSON 请求
- POST /api/search/upload - 文件上传
- GET /api/search/status - 服务状态
- POST /api/search/rebuild-cache - 重建缓存
"""

from fastapi import APIRouter, HTTPException, Query, UploadFile, File, Form
from typing import List, Dict, Any, Optional
import io
import logging
import time

from app.schemas.unified import (
    SearchMode,
    EncryptionMode,
    UnifiedSearchRequest,
    UnifiedSearchResponse,
    BaseSearchResult,
    TagToImageResult,
    ImageToTagResult,
    ImageToImageResult,
    HitStats,
    EncryptionInfo,
    UnifiedStatusResponse,
    ServiceStatus,
    DCMH_DATASETS,
    CIR_DATASETS,
    ALL_DATASETS
)
from app.services.base_search_service import SearchServiceFactory, SearchContext
from app.services.key_manager import get_key_manager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/search", tags=["unified-search"])


def _validate_dataset(dataset: str, mode: SearchMode) -> None:
    """
    验证数据集与检索模式的兼容性。

    参数：
        dataset: 数据集名称
        mode: 检索模式

    抛出：
        HTTPException: 数据集与模式不兼容
    """
    if mode in [SearchMode.TAG_TO_IMAGE, SearchMode.IMAGE_TO_TAG]:
        if dataset not in DCMH_DATASETS:
            raise HTTPException(
                status_code=400,
                detail=f"数据集 {dataset} 不支持 {mode.value} 模式，"
                       f"支持的数据集：{DCMH_DATASETS}"
            )
    elif mode == SearchMode.IMAGE_TO_IMAGE:
        if dataset not in CIR_DATASETS:
            raise HTTPException(
                status_code=400,
                detail=f"数据集 {dataset} 不支持 {mode.value} 模式，"
                       f"支持的数据集：{CIR_DATASETS}"
            )


@router.post("", response_model=UnifiedSearchResponse)
async def unified_search(request: UnifiedSearchRequest) -> UnifiedSearchResponse:
    """
    统一检索端点（JSON 请求）。

    支持三种检索模式：
    - tag_to_image: 标签搜图（DCMH），需要 tag_indices
    - image_to_tag: 图搜标签（DCMH），需要 query_image (Base64)
    - image_to_image: 图搜图（CIR），需要 query_image (Base64)

    支持两种加密模式：
    - plaintext: 明文检索
    - encrypted: 加密检索
    """
    try:
        start_time = time.time()

        # 验证数据集
        _validate_dataset(request.dataset, request.mode)

        # 获取服务
        service = SearchServiceFactory.get_service(request.dataset)
        if service is None:
            raise HTTPException(
                status_code=400,
                detail=f"不支持的数据集：{request.dataset}"
            )

        # 构建上下文
        context = SearchContext(
            mode=request.mode,
            encryption=request.encryption,
            dataset=request.dataset,
            top_k=request.top_k,
            tag_indices=request.tag_indices,
            query_image_base64=request.query_image
        )

        # 执行检索
        if request.mode == SearchMode.TAG_TO_IMAGE:
            # 标签搜图
            if not request.tag_indices:
                return UnifiedSearchResponse(
                    success=False,
                    mode=request.mode,
                    encryption=request.encryption,
                    dataset=request.dataset,
                    total_results=0,
                    search_time_ms=0,
                    message="请提供 tag_indices"
                )

            results, metadata = await service.search_by_tags(request.tag_indices, context)

            return UnifiedSearchResponse(
                success=True,
                mode=request.mode,
                encryption=request.encryption,
                dataset=request.dataset,
                results=results,
                total_results=len(results),
                search_time_ms=metadata.get("search_time_ms", 0),
                hit_stats=metadata.get("hit_stats"),
                query_hash_code=metadata.get("query_hash_code"),
                encryption_info=metadata.get("encryption_info")
            )

        elif request.mode == SearchMode.IMAGE_TO_TAG:
            # 图搜标签
            if not request.query_image:
                return UnifiedSearchResponse(
                    success=False,
                    mode=request.mode,
                    encryption=request.encryption,
                    dataset=request.dataset,
                    total_results=0,
                    search_time_ms=0,
                    message="请提供 query_image"
                )

            # 解码 Base64 图像
            import base64
            from PIL import Image

            base64_str = request.query_image
            if ',' in base64_str:
                base64_str = base64_str.split(',')[1]

            image_data = base64.b64decode(base64_str)

            results, metadata = await service.search_by_image(image_data, context)

            return UnifiedSearchResponse(
                success=True,
                mode=request.mode,
                encryption=request.encryption,
                dataset=request.dataset,
                tag_results=results,
                total_results=len(results),
                search_time_ms=metadata.get("search_time_ms", 0),
                hit_stats=metadata.get("hit_stats"),
                query_hash_code=metadata.get("query_hash_code"),
                encryption_info=metadata.get("encryption_info")
            )

        elif request.mode == SearchMode.IMAGE_TO_IMAGE:
            # 图搜图
            if not request.query_image:
                return UnifiedSearchResponse(
                    success=False,
                    mode=request.mode,
                    encryption=request.encryption,
                    dataset=request.dataset,
                    total_results=0,
                    search_time_ms=0,
                    message="请提供 query_image"
                )

            # 解码 Base64 图像
            import base64

            base64_str = request.query_image
            if ',' in base64_str:
                base64_str = base64_str.split(',')[1]

            image_data = base64.b64decode(base64_str)

            results, metadata = await service.search_by_image(image_data, context)

            return UnifiedSearchResponse(
                success=True,
                mode=request.mode,
                encryption=request.encryption,
                dataset=request.dataset,
                results=results,
                total_results=len(results),
                search_time_ms=metadata.get("search_time_ms", 0),
                hit_stats=metadata.get("hit_stats"),
                encryption_info=metadata.get("encryption_info")
            )

        else:
            raise HTTPException(
                status_code=400,
                detail=f"不支持的检索模式：{request.mode}"
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"检索失败：{e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/upload", response_model=UnifiedSearchResponse)
async def unified_search_upload(
    image: UploadFile = File(..., description="查询图像"),
    mode: SearchMode = Form(default=SearchMode.IMAGE_TO_IMAGE, description="检索模式"),
    encryption: EncryptionMode = Form(default=EncryptionMode.ENCRYPTED, description="加密模式"),
    dataset: str = Form(default="roxford5k", description="数据集名称"),
    top_k: int = Form(default=10, ge=1, le=100, description="返回结果数量")
) -> UnifiedSearchResponse:
    """
    统一检索端点（文件上传）。

    用于处理 multipart/form-data 文件上传。

    参数：
        image: 查询图像文件
        mode: 检索模式（image_to_tag 或 image_to_image）
        encryption: 加密模式
        dataset: 数据集名称
        top_k: 返回结果数量
    """
    try:
        start_time = time.time()

        # 验证数据集
        _validate_dataset(dataset, mode)

        # 标签搜图不支持文件上传
        if mode == SearchMode.TAG_TO_IMAGE:
            raise HTTPException(
                status_code=400,
                detail="tag_to_image 模式不支持文件上传，请使用 JSON 请求并提供 tag_indices"
            )

        # 获取服务
        service = SearchServiceFactory.get_service(dataset)
        if service is None:
            raise HTTPException(
                status_code=400,
                detail=f"不支持的数据集：{dataset}"
            )

        # 读取图像数据
        image_data = await image.read()

        # 构建上下文
        context = SearchContext(
            mode=mode,
            encryption=encryption,
            dataset=dataset,
            top_k=top_k,
            query_image_data=image_data
        )

        # 执行检索
        results, metadata = await service.search_by_image(image_data, context)

        # 构建响应
        if mode == SearchMode.IMAGE_TO_TAG:
            return UnifiedSearchResponse(
                success=True,
                mode=mode,
                encryption=encryption,
                dataset=dataset,
                tag_results=results,
                total_results=len(results),
                search_time_ms=metadata.get("search_time_ms", 0),
                hit_stats=metadata.get("hit_stats"),
                encryption_info=metadata.get("encryption_info")
            )
        else:
            return UnifiedSearchResponse(
                success=True,
                mode=mode,
                encryption=encryption,
                dataset=dataset,
                results=results,
                total_results=len(results),
                search_time_ms=metadata.get("search_time_ms", 0),
                hit_stats=metadata.get("hit_stats"),
                encryption_info=metadata.get("encryption_info")
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"检索失败：{e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status", response_model=UnifiedStatusResponse)
async def get_status(
    dataset: str = Query(default="flickr25k", description="数据集名称")
) -> UnifiedStatusResponse:
    """
    获取检索服务状态。

    返回 DCMH 和 CIR 服务的状态信息。
    """
    key_manager = get_key_manager()

    dcmh_status = None
    cir_status = None

    # 获取 DCMH 状态
    if dataset in DCMH_DATASETS:
        from app.services.dcmh_search_service import get_dcmh_search_service
        service = get_dcmh_search_service()
        dcmh_status = service.get_status(dataset)

    # 获取 CIR 状态
    if dataset in CIR_DATASETS:
        from app.services.cir_search_service import get_cir_search_service
        service = get_cir_search_service()
        cir_status = service.get_status(dataset)

    return UnifiedStatusResponse(
        success=True,
        dcmh_status=dcmh_status,
        cir_status=cir_status,
        key_manager_status=key_manager.get_status()
    )


@router.post("/rebuild-cache")
async def rebuild_cache(
    dataset: str = Query(default="flickr25k", description="数据集名称")
) -> Dict[str, Any]:
    """
    重建检索缓存。

    参数：
        dataset: 数据集名称
    """
    if dataset not in ALL_DATASETS:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的数据集：{dataset}，支持的数据集：{ALL_DATASETS}"
        )

    service = SearchServiceFactory.get_service(dataset)
    if service is None:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的数据集：{dataset}"
        )

    success = await service.rebuild_index(dataset)

    return {
        "success": success,
        "dataset": dataset,
        "message": f"缓存重建{'成功' if success else '失败'}"
    }


@router.get("/datasets")
async def get_supported_datasets() -> Dict[str, Any]:
    """
    获取支持的数据集列表。

    返回 DCMH 和 CIR 支持的数据集。
    """
    return {
        "dcmh_datasets": DCMH_DATASETS,
        "cir_datasets": CIR_DATASETS,
        "all_datasets": ALL_DATASETS,
        "modes": {
            "tag_to_image": {
                "description": "标签搜图",
                "supported_datasets": DCMH_DATASETS
            },
            "image_to_tag": {
                "description": "图搜标签",
                "supported_datasets": DCMH_DATASETS
            },
            "image_to_image": {
                "description": "图搜图",
                "supported_datasets": CIR_DATASETS
            }
        }
    }