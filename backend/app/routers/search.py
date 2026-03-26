"""
搜索 API 路由

提供隐私保护跨模态检索端点。
支持多数据集切换。

关键改进：
- 双向检索：label_to_image 和 image_to_label
- 使用正确的数据库（图像哈希码 vs 文本哈希码）
- 正确的图像预处理
"""

from fastapi import APIRouter, HTTPException, Query
import numpy as np
import time
from typing import List, Optional
import torch
import logging

from app.schemas.search import (
    SearchRequest,
    SearchResponse,
    SearchResult,
    EncryptionInfo,
    ImageToLabelResult,
    DCMH_DATASETS
)
from app.services.aspe_service import get_aspe_service
from app.services.dcmh_service import get_dcmh_service
from app.services.dataset_service import get_dataset_service
from app.services.hash_cache_service import get_hash_cache_service, DCMH_CACHE_DIR

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/search", tags=["search"])

# 缓存初始化标志（按数据集）
_cache_initialized: dict = {}
_current_dcmh_dataset: str = "flickr25k"


async def ensure_cache_initialized(dataset: str = "flickr25k"):
    """
    确保缓存已初始化。

    参数：
        dataset: 数据集名称（默认 flickr25k）
    """
    global _cache_initialized, _current_dcmh_dataset

    if _cache_initialized.get(dataset, False):
        return

    try:
        hash_cache = get_hash_cache_service()
        aspe_service = get_aspe_service()
        dcmh_service = get_dcmh_service(dataset=dataset)
        dataset_service = get_dataset_service(dataset_name=dataset)

        # 检查模型是否已加载
        if not dcmh_service.is_loaded():
            logger.warning("DCMH 模型未加载，将使用随机哈希码")

        # 尝试加载完整的数据库缓存
        image_codes, text_codes, labels = hash_cache.load_full_database(dataset)

        if image_codes is not None:
            hash_cache.database_codes = image_codes
            hash_cache.database_text_codes = text_codes
            hash_cache.database_labels = labels

            # 加密数据库
            if hash_cache.dcmh_cache_exists(dataset, "encrypted"):
                encrypted = hash_cache.load_dcmh_encrypted(dataset)
                if encrypted is not None:
                    hash_cache.encrypted_database = encrypted
                    aspe_service.encrypted_database = encrypted
                    aspe_service.database_codes = image_codes
                    aspe_service.database_labels = labels
            else:
                # 加密数据库
                encrypted = aspe_service.encrypt_database(image_codes, labels)
                hash_cache.encrypted_database = encrypted

            aspe_service.database_codes = image_codes
            aspe_service.database_labels = labels

            logger.info(f"[DCMH] 缓存已加载 ({dataset})：图像 {image_codes.shape[0]} 条，"
                       f"文本 {text_codes.shape[0] if text_codes is not None else 0} 条")

        _cache_initialized[dataset] = True
        _current_dcmh_dataset = dataset

    except Exception as e:
        logger.error(f"缓存初始化失败：{e}")
        _cache_initialized[dataset] = True


@router.post("", response_model=SearchResponse)
async def search(request: SearchRequest):
    """
    执行跨模态检索。

    支持：
    - label_to_image: 标签→图像检索（使用图像哈希码数据库）
    - image_to_label: 图像→标签检索（使用文本哈希码数据库）
    - 多数据集切换（flickr25k, nuswide）
    - 加密/明文检索
    """
    try:
        start_time = time.time()

        # 验证数据集
        if request.dataset not in DCMH_DATASETS:
            raise HTTPException(
                status_code=400,
                detail=f"不支持的数据集：{request.dataset}，支持的数据集：{DCMH_DATASETS}"
            )

        # 确保缓存初始化
        await ensure_cache_initialized(request.dataset)

        aspe_service = get_aspe_service()
        dcmh_service = get_dcmh_service(dataset=request.dataset)
        hash_cache = get_hash_cache_service()
        dataset_service = get_dataset_service(dataset_name=request.dataset)

        # 检查数据库是否已构建
        if hash_cache.database_codes is None:
            # 如果没有缓存，尝试构建
            if dcmh_service.is_loaded():
                await _build_database_from_real_data(request.dataset)
            else:
                # 回退到演示模式
                await _build_demo_database(request.dataset)

        # 生成查询哈希码
        query_code_np = None
        database_codes = None  # 用于检索的数据库
        query_label_names = []  # 查询标签名称列表

        if request.query_type == "label_to_image" or request.query_type == "label":
            # ========== 标签→图像检索 ==========
            if not request.label_indices:
                return SearchResponse(
                    success=False,
                    query_type="label_to_image",
                    label_indices=[],
                    results=[],
                    total_results=0,
                    search_time_ms=0
                )

            # 前端返回的是 YAll 索引，直接构建 1386 维 multi-hot 向量
            # 不从 common_tags.txt 加载，使用固定维度
            label_dim = 1386
            label_vector = np.zeros(label_dim, dtype=np.float32)
            for idx in request.label_indices:
                if 0 <= idx < label_dim:
                    label_vector[idx] = 1.0

            # 生成文本哈希码
            text_tensor = torch.from_numpy(label_vector).unsqueeze(0).float()
            query_code = dcmh_service.generate_text_code_single(text_tensor)
            query_code_np = query_code.cpu().numpy().squeeze()

            # 使用图像哈希码数据库进行检索
            database_codes = hash_cache.database_codes

            logger.info(f"[搜索] 标签→图像：label_indices={request.label_indices}, "
                       f"query_code shape={query_code_np.shape}, db shape={database_codes.shape if database_codes is not None else None}")

        elif request.query_type == "image_to_label":
            # ========== 图像→标签检索 ==========
            if request.query_image:
                # 处理 base64 图像
                query_code_np = _process_base64_image(
                    request.query_image, dcmh_service, dataset_service
                )
            else:
                return SearchResponse(
                    success=False,
                    query_type="image_to_label",
                    label_indices=[],
                    results=[],
                    total_results=0,
                    search_time_ms=0,
                    message="请提供查询图像"
                )

            # 使用文本哈希码数据库进行检索
            database_codes = hash_cache.database_text_codes

            if database_codes is None:
                return SearchResponse(
                    success=False,
                    query_type="image_to_label",
                    label_indices=[],
                    results=[],
                    total_results=0,
                    search_time_ms=0,
                    message="文本哈希码数据库未初始化"
                )

            logger.info(f"[搜索] 图像→标签：query_code shape={query_code_np.shape}, "
                       f"db shape={database_codes.shape}")

        else:
            return SearchResponse(
                success=False,
                query_type=request.query_type,
                label_indices=request.label_indices,
                results=[],
                total_results=0,
                search_time_ms=0,
                message=f"不支持的查询类型：{request.query_type}，支持的类型：label_to_image, image_to_label"
            )

        # 检查数据库
        if database_codes is None:
            return SearchResponse(
                success=False,
                query_type=request.query_type,
                label_indices=request.label_indices,
                results=[],
                total_results=0,
                search_time_ms=0,
                message="检索数据库未初始化"
            )

        # 执行检索
        distances = None
        encryption_info = EncryptionInfo(
            method="ASPE Scheme 1",
            query_encrypted=False,
            database_encrypted=False,
            security_level=2,
            bit_dim=dcmh_service.bit_dim
        )

        if request.use_encrypted and aspe_service.encrypted_database is not None:
            # ASPE 加密检索
            encrypted_query = aspe_service.generate_trapdoor(query_code_np.reshape(1, -1))
            distances = aspe_service.compute_ciphertext_distances(encrypted_query)

            encryption_info.query_encrypted = True
            encryption_info.database_encrypted = True

        else:
            # 明文检索
            distances = aspe_service._plaintext_hamming_distance(
                query_code_np.reshape(1, -1), database_codes
            )

        # 获取Top-K结果
        distances_flat = distances.squeeze()
        top_k_indices = np.argsort(distances_flat)[:request.top_k]

        # 获取数据库索引
        dataset_service.load_data()
        _, _, retrieval_indices = dataset_service.get_data_split_indices()

        # 获取检索库标签
        retrieval_labels = dataset_service.get_yall(retrieval_indices)

        # 根据查询类型构建结果
        results: List[SearchResult] = []
        label_results: List[ImageToLabelResult] = []

        if request.query_type == "image_to_label":
            # ========== 图像→标签检索：返回相似图像的标签 ==========
            for rank, idx in enumerate(top_k_indices):
                # 获取实际图像索引
                actual_idx = retrieval_indices[idx] if idx < len(retrieval_indices) else idx
                image_id = int(actual_idx)

                # 获取该图像的标签索引列表 (YAll 索引)
                image_labels = []
                if retrieval_labels is not None and idx < len(retrieval_labels):
                    label_row = retrieval_labels[idx]
                    image_labels = np.where(label_row > 0)[0].tolist()

                # 获取标签名称
                label_names = dataset_service.get_label_names_from_yall_indices(image_labels[:20])

                label_results.append(ImageToLabelResult(
                    rank=rank + 1,
                    image_id=image_id,
                    labels=image_labels[:20] if image_labels else [],
                    label_names=label_names,
                    score=float(1.0 / (1.0 + distances_flat[idx])),
                    distance=float(distances_flat[idx])
                ))

            # 图像→标签检索的统计
            hit_stats = None
        else:
            # ========== 标签→图像检索：返回图像结果 ==========
            # 前端发送的 label_indices 已经是 YAll 索引，直接使用
            query_yall_indices = set(request.label_indices)

            # 获取查询标签名称
            query_label_names = dataset_service.get_label_names_from_yall_indices(list(query_yall_indices))

            # 统计命中数
            total_hits = 0
            total_query_labels = len(query_yall_indices)

            for rank, idx in enumerate(top_k_indices):
                # 获取实际图像索引
                actual_idx = retrieval_indices[idx] if idx < len(retrieval_indices) else idx
                image_id = int(actual_idx)

                # 获取该图像的标签索引列表 (YAll 索引)
                image_labels = []
                if retrieval_labels is not None and idx < len(retrieval_labels):
                    label_row = retrieval_labels[idx]
                    image_labels = np.where(label_row > 0)[0].tolist()

                # 计算命中标签（检索结果中包含的查询标签）
                hit_labels = [l for l in image_labels if l in query_yall_indices]

                # 获取标签名称
                label_names = dataset_service.get_label_names_from_yall_indices(image_labels[:20])
                hit_label_names = dataset_service.get_label_names_from_yall_indices(hit_labels)

                # 统计命中
                if hit_labels:
                    total_hits += 1

                # 获取结果哈希码
                result_hash_code = None
                if database_codes is not None and idx < len(database_codes):
                    result_hash_code = database_codes[idx].tolist()

                # 生成图像 URL（使用 clean_id 映射获取原始图像 ID）
                original_id = dataset_service.get_original_image_id(image_id)
                thumbnail_url = f"/flickr-images/im{original_id + 1}.jpg"

                results.append(SearchResult(
                    rank=rank + 1,
                    image_id=image_id,
                    score=float(1.0 / (1.0 + distances_flat[idx])),
                    distance=float(distances_flat[idx]),
                    labels=image_labels[:20] if image_labels else [],
                    label_names=label_names,
                    hit_labels=hit_labels,
                    hit_label_names=hit_label_names,
                    thumbnail_url=thumbnail_url,
                    hash_code=result_hash_code
                ))

            # 计算命中率统计
            hit_stats = {
                "total_results": len(results),
                "hits": total_hits,
                "hit_rate": total_hits / len(results) if results else 0,
                "query_label_count": total_query_labels,
                "query_labels": list(query_yall_indices),
                "query_label_names": query_label_names
            }

        search_time_ms = (time.time() - start_time) * 1000

        return SearchResponse(
            success=True,
            query_type=request.query_type,
            label_indices=request.label_indices,
            query_label_names=query_label_names if request.query_type != "image_to_label" else [],
            results=results,
            label_results=label_results,
            total_results=len(results) + len(label_results),
            search_time_ms=search_time_ms,
            hit_stats=hit_stats,
            encryption_info=encryption_info,
            query_hash_code=query_code_np.tolist() if query_code_np is not None else None
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


async def _build_database_from_real_data(dataset: str = "flickr25k"):
    """使用真实数据构建数据库。"""
    hash_cache = get_hash_cache_service()
    dcmh_service = get_dcmh_service(dataset=dataset)
    dataset_service = get_dataset_service(dataset_name=dataset)
    aspe_service = get_aspe_service()

    try:
        # 构建完整的数据库缓存
        image_codes, text_codes, labels = hash_cache.build_full_database_cache(
            dcmh_service, dataset_service, batch_size=32, force_rebuild=False, dataset=dataset
        )

        # 构建加密缓存
        hash_cache.build_encrypted_cache(aspe_service, force_rebuild=False, dataset=dataset)

        logger.info(f"真实数据库构建完成 ({dataset})：图像 {image_codes.shape}, 文本 {text_codes.shape}")

    except Exception as e:
        logger.error(f"构建真实数据库失败：{e}")
        await _build_demo_database(dataset)


async def _build_demo_database(dataset: str = "flickr25k"):
    """构建演示数据库。"""
    aspe_service = get_aspe_service()
    dcmh_service = get_dcmh_service(dataset=dataset)

    # 构建演示数据库
    num_images = 100
    demo_hash_codes = np.sign(np.random.randn(num_images, dcmh_service.bit_dim))

    # 生成随机标签
    labels = np.random.randint(0, 2, (num_images, 10)).astype(np.float32)

    # 加密
    aspe_service.encrypt_database(demo_hash_codes, labels)

    # 缓存数据库代码用于明文检索
    aspe_service.database_codes = demo_hash_codes

    logger.info(f"演示数据库构建完成 ({dataset})：{demo_hash_codes.shape}")


def _generate_demo_query_code(bit_dim: int) -> np.ndarray:
    """生成演示查询哈希码。"""
    return np.sign(np.random.randn(1, bit_dim))


def _process_base64_image(base64_str: str, dcmh_service, dataset_service) -> np.ndarray:
    """
    处理 base64 编码的图像并生成哈希码。

    参数：
        base64_str: base64 编码的图像字符串
        dcmh_service: DCMH 服务实例
        dataset_service: 数据集服务实例

    返回：
        哈希码数组 [bit_dim]
    """
    import base64
    from io import BytesIO
    from PIL import Image
    from app.services.dcmh_service import preprocess_image_for_inference

    # 解码 base64
    if ',' in base64_str:
        base64_str = base64_str.split(',')[1]

    image_data = base64.b64decode(base64_str)
    image = Image.open(BytesIO(image_data))

    # 预处理图像（使用正确的预处理）
    image_tensor = preprocess_image_for_inference(image).unsqueeze(0)

    # 生成哈希码
    hash_code = dcmh_service.generate_image_code_single(image_tensor)
    return hash_code.cpu().numpy().squeeze()


@router.get("/demo")
async def demo_search(
    label_indices: str = "0,5,10",
    top_k: int = 5,
    dataset: str = Query(default="flickr25k", description="数据集名称")
):
    """
    演示搜索（简化版本）。

    参数：
    - label_indices: 逗号分隔的标签索引，例如 "0,5,10"
    - top_k: 返回的结果数量
    - dataset: 数据集名称
    """
    try:
        # 解析标签索引
        indices = [int(x.strip()) for x in label_indices.split(",") if x.strip().isdigit()]

        # 创建搜索请求
        request = SearchRequest(
            query_type="label",
            label_indices=indices,
            dataset=dataset,
            top_k=top_k,
            use_encrypted=True
        )

        # 执行搜索
        return await search(request)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status")
async def search_status(dataset: str = Query(default="flickr25k", description="数据集名称")):
    """获取搜索服务状态。"""
    hash_cache = get_hash_cache_service()
    dcmh_service = get_dcmh_service(dataset=dataset)
    aspe_service = get_aspe_service()

    return {
        "cache_info": hash_cache.get_cache_info(),
        "dcmh_status": dcmh_service.get_status(),
        "aspe_status": aspe_service.get_status(),
        "cache_initialized": _cache_initialized.get(dataset, False),
        "current_dataset": _current_dcmh_dataset
    }


@router.post("/rebuild-cache")
async def rebuild_cache(
    dataset: str = Query(default="flickr25k", description="数据集名称")
):
    """
    重建哈希码缓存。

    参数：
        dataset: 数据集名称（flickr25k, nuswide）
    """
    global _cache_initialized, _current_dcmh_dataset

    # 验证数据集
    if dataset not in DCMH_DATASETS:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的数据集：{dataset}，支持的数据集：{DCMH_DATASETS}"
        )

    _cache_initialized[dataset] = False

    try:
        hash_cache = get_hash_cache_service()
        dcmh_service = get_dcmh_service(dataset=dataset)
        dataset_service = get_dataset_service(dataset_name=dataset)
        aspe_service = get_aspe_service()

        if not dcmh_service.is_loaded():
            return {"success": False, "message": "DCMH 模型未加载"}

        # 重建完整的数据库缓存
        image_codes, text_codes, labels = hash_cache.build_full_database_cache(
            dcmh_service, dataset_service, batch_size=32, force_rebuild=True, dataset=dataset
        )

        hash_cache.build_encrypted_cache(aspe_service, force_rebuild=True, dataset=dataset)

        _cache_initialized[dataset] = True
        _current_dcmh_dataset = dataset

        return {
            "success": True,
            "message": f"缓存重建完成 ({dataset})",
            "dataset": dataset,
            "database_size": image_codes.shape[0],
            "text_database_size": text_codes.shape[0]
        }

    except Exception as e:
        return {"success": False, "message": str(e)}