"""
搜索 API 路由

提供隐私保护跨模态检索端点。
支持多数据集切换。

关键改进：
- 双向检索：tag_to_image 和 image_to_tag
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
    ImageToTagResult,
    DCMH_DATASETS,
    HitStats
)
from app.services.aspe_service import get_aspe_service
from app.services.dcmh_service import get_dcmh_service
from app.services.dataset_service import get_dataset_service, DATASET_CONFIGS
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

        # 尝试加载完整的数据库缓存（返回 YAll）
        image_codes, text_codes, yall = hash_cache.load_dcmh_yall(dataset)

        if image_codes is not None:
            hash_cache.database_codes = image_codes
            hash_cache.database_text_codes = text_codes
            hash_cache.database_tags = yall  # YAll 用于检索结果显示

            # 加载 LAll 类别标签（用于 mAP 计算，与加密无关）
            lall = hash_cache.load_dcmh_lall(dataset)

            # 检查是否已有加密缓存
            encrypted_image = hash_cache.load_dcmh_encrypted(dataset)
            encrypted_text = hash_cache.load_dcmh_encrypted_text(dataset)

            if encrypted_image is not None:
                hash_cache.encrypted_database = encrypted_image
                aspe_service.encrypted_database = encrypted_image
            else:
                # 加密图像哈希码（用于标签→图像检索）
                # 只需要哈希码本身，LAll 仅用于 mAP 计算
                encrypted_image = aspe_service.encrypt_database(image_codes, lall)
                hash_cache.encrypted_database = encrypted_image
                hash_cache.save_dcmh_encrypted(encrypted_image, dataset)

            if encrypted_text is not None:
                hash_cache.encrypted_text_database = encrypted_text
                aspe_service.encrypted_text_database = encrypted_text
            else:
                # 加密文本哈希码（用于图像→标签检索）
                # 只需要哈希码本身，不需要标签
                encrypted_text = aspe_service.encrypt_text_database(text_codes)
                hash_cache.encrypted_text_database = encrypted_text
                aspe_service.encrypted_text_database = encrypted_text
                hash_cache.save_dcmh_encrypted_text(encrypted_text, dataset)

            aspe_service.database_codes = image_codes
            if lall is not None:
                aspe_service.database_labels = lall  # LAll 用于 mAP 计算

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
    - tag_to_image: 标签→图像检索（使用图像哈希码数据库）
    - image_to_tag: 图像→标签检索（使用文本哈希码数据库）
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
        query_tag_names = []  # 查询标签名称列表

        if request.query_type == "tag_to_image" or request.query_type == "tag":
            # ========== 标签→图像检索 ==========
            if not request.tag_indices:
                return SearchResponse(
                    success=False,
                    query_type="tag_to_image",
                    tag_indices=[],
                    results=[],
                    total_results=0,
                    search_time_ms=0
                )

            # 前端返回的是 YAll 索引，直接构建 1386 维 multi-hot 向量
            # 不从 common_tags.txt 加载，使用固定维度
            tag_dim = 1386
            tag_vector = np.zeros(tag_dim, dtype=np.float32)
            for idx in request.tag_indices:
                if 0 <= idx < tag_dim:
                    tag_vector[idx] = 1.0

            # 生成文本哈希码
            text_tensor = torch.from_numpy(tag_vector).unsqueeze(0).float()
            query_code = dcmh_service.generate_text_code_single(text_tensor)
            query_code_np = query_code.cpu().numpy().squeeze()

            # 使用图像哈希码数据库进行检索
            database_codes = hash_cache.database_codes

            logger.info(f"[搜索] 标签→图像：tag_indices={request.tag_indices}, "
                       f"query_code shape={query_code_np.shape}, db shape={database_codes.shape if database_codes is not None else None}")

        elif request.query_type == "image_to_tag":
            # ========== 图像→标签检索 ==========
            if request.query_image:
                # 处理 base64 图像
                query_code_np = _process_base64_image(
                    request.query_image, dcmh_service, dataset_service
                )
            else:
                return SearchResponse(
                    success=False,
                    query_type="image_to_tag",
                    tag_indices=[],
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
                    query_type="image_to_tag",
                    tag_indices=[],
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
                tag_indices=request.tag_indices,
                results=[],
                total_results=0,
                search_time_ms=0,
                message=f"不支持的查询类型：{request.query_type}，支持的类型：tag_to_image, image_to_tag"
            )

        # 检查数据库
        if database_codes is None:
            return SearchResponse(
                success=False,
                query_type=request.query_type,
                tag_indices=request.tag_indices,
                results=[],
                total_results=0,
                search_time_ms=0,
                message="检索数据库未初始化"
            )

        # 执行检索
        distances = None
        # 从 aspe_service 获取正确的 scheme 信息
        aspe_status = aspe_service.get_status()
        encryption_info = EncryptionInfo(
            method=aspe_status.get("scheme", "ASPE Scheme 2"),
            query_encrypted=False,
            database_encrypted=False,
            security_level=aspe_status.get("security_level", 3),
            bit_dim=dcmh_service.bit_dim
        )

        if request.use_encrypted:
            # ASPE 加密检索
            encrypted_query = aspe_service.generate_trapdoor(query_code_np.reshape(1, -1))

            # 根据查询类型选择正确的加密数据库
            if request.query_type == "image_to_tag":
                # 图像→标签：使用加密的文本哈希码数据库
                if aspe_service.encrypted_text_database is not None:
                    distances = aspe_service.compute_ciphertext_distances(
                        encrypted_query, aspe_service.encrypted_text_database
                    )
                    encryption_info.query_encrypted = True
                    encryption_info.database_encrypted = True
                else:
                    # 回退到明文检索
                    logger.warning("[搜索] 加密文本数据库未初始化，回退到明文检索")
                    distances = aspe_service._plaintext_hamming_distance(
                        query_code_np.reshape(1, -1), database_codes
                    )
            else:
                # 标签→图像：使用加密的图像哈希码数据库
                if aspe_service.encrypted_database is not None:
                    distances = aspe_service.compute_ciphertext_distances(encrypted_query)
                    encryption_info.query_encrypted = True
                    encryption_info.database_encrypted = True
                else:
                    # 回退到明文检索
                    logger.warning("[搜索] 加密图像数据库未初始化，回退到明文检索")
                    distances = aspe_service._plaintext_hamming_distance(
                        query_code_np.reshape(1, -1), database_codes
                    )

        else:
            # 明文检索
            distances = aspe_service._plaintext_hamming_distance(
                query_code_np.reshape(1, -1), database_codes
            )

        # 获取Top-K结果
        distances_flat = distances.squeeze()
        # 使用统一的排序逻辑：四舍五入 + lexsort 确保相同距离时的确定性排序
        distances_rounded = np.round(distances_flat, decimals=10)
        sorted_indices = np.lexsort((np.arange(len(distances_rounded)), distances_rounded))
        top_k_indices = sorted_indices[:request.top_k]

        # 获取数据库索引
        dataset_service.load_data()
        _, _, retrieval_indices = dataset_service.get_data_split_indices()

        # 获取检索库标签
        retrieval_tags = dataset_service.get_yall(retrieval_indices)

        # 加载 LAll 类别标签（用于类别命中率计算）
        retrieval_lall = hash_cache.load_dcmh_lall(request.dataset)

        # 根据查询类型构建结果
        results: List[SearchResult] = []
        tag_results: List[ImageToTagResult] = []

        if request.query_type == "image_to_tag":
            # ========== 图像→标签检索：返回相似图像的标签 ==========
            # 统计类别分布
            category_counter = {}

            for rank, idx in enumerate(top_k_indices):
                # 获取实际图像索引
                actual_idx = retrieval_indices[idx] if idx < len(retrieval_indices) else idx
                image_id = int(actual_idx)

                # 获取该图像的标签索引列表 (YAll 索引)
                image_tags = []
                if retrieval_tags is not None and idx < len(retrieval_tags):
                    tag_row = retrieval_tags[idx]
                    image_tags = np.where(tag_row > 0)[0].tolist()

                # 获取标签名称
                tag_names = dataset_service.get_tag_names_from_yall_indices(image_tags[:20])

                # 获取类别名称
                result_category_names = []
                if retrieval_lall is not None and idx < len(retrieval_lall):
                    result_lall = retrieval_lall[idx]
                    result_category_indices = np.where(result_lall > 0)[0].tolist()
                    result_category_names = dataset_service.get_category_names_from_lall_indices(result_category_indices)

                    # 统计类别分布
                    for cat in result_category_names:
                        category_counter[cat] = category_counter.get(cat, 0) + 1

                # 生成来源图像缩略图 URL
                config = DATASET_CONFIGS.get(request.dataset, {})
                if config.get('type') == 'raw':
                    # 使用静态文件 URL
                    original_id = dataset_service.get_original_image_id(image_id)
                    thumbnail_url = f"/flickr-images/im{original_id + 1}.jpg"
                else:
                    # 使用 API 端点
                    thumbnail_url = f"/api/images/{image_id}?format=image&dataset={request.dataset}"

                tag_results.append(ImageToTagResult(
                    rank=rank + 1,
                    image_id=image_id,
                    tags=image_tags[:20] if image_tags else [],
                    tag_names=tag_names,
                    score=float(1.0 / (1.0 + distances_flat[idx])),
                    distance=float(distances_flat[idx]),
                    thumbnail_url=thumbnail_url,
                    category_names=result_category_names
                ))

            # 图像→标签检索的统计（类别分布）
            hit_stats = {
                "total_results": len(tag_results),
                "category_distribution": category_counter,
                "query_type": "image_to_tag"
            }
        else:
            # ========== 标签→图像检索：返回图像结果 ==========
            # 前端发送的 tag_indices 已经是 YAll 索引，直接使用
            query_yall_indices = set(request.tag_indices)

            # 获取查询标签名称
            query_tag_names = dataset_service.get_tag_names_from_yall_indices(list(query_yall_indices))

            # 根据查询的 YAll 索引获取对应的 LAll 向量
            # 找到包含这些 YAll 标签的图像，取其 LAll 的并集
            query_lall_vector = None
            if retrieval_lall is not None and retrieval_tags is not None:
                # 找到包含任意查询标签的图像
                mask = np.zeros(len(retrieval_tags), dtype=bool)
                for tag_idx in query_yall_indices:
                    if tag_idx < retrieval_tags.shape[1]:
                        mask |= (retrieval_tags[:, tag_idx] > 0)

                # 获取这些图像的 LAll，取并集
                matching_lall = retrieval_lall[mask]
                if len(matching_lall) > 0:
                    query_lall_vector = matching_lall.max(axis=0)

            # 统计命中数
            total_tag_hits = 0
            total_category_hits = 0

            for rank, idx in enumerate(top_k_indices):
                # 获取实际图像索引
                actual_idx = retrieval_indices[idx] if idx < len(retrieval_indices) else idx
                image_id = int(actual_idx)

                # 获取该图像的标签索引列表 (YAll 索引)
                image_tags = []
                if retrieval_tags is not None and idx < len(retrieval_tags):
                    tag_row = retrieval_tags[idx]
                    image_tags = np.where(tag_row > 0)[0].tolist()

                # 计算命中标签（检索结果中包含的查询标签）- YAll 命中
                hit_tags = [t for t in image_tags if t in query_yall_indices]
                tag_hit = len(hit_tags) > 0

                # 计算 LAll 类别命中
                category_hit = False
                hit_category_names = []
                result_category_names = []
                if retrieval_lall is not None and query_lall_vector is not None and idx < len(retrieval_lall):
                    # 如果结果图像的 LAll 与查询 LAll 有交集，则类别命中
                    result_lall = retrieval_lall[idx]
                    # 检查是否有共同的类别
                    category_hit = np.any((result_lall > 0) & (query_lall_vector > 0))

                    # 获取结果图像的所有类别索引
                    result_category_indices = np.where(result_lall > 0)[0].tolist()
                    result_category_names = dataset_service.get_category_names_from_lall_indices(result_category_indices)

                    # 获取命中的类别索引（与查询 LAll 有交集的类别）
                    if category_hit:
                        hit_category_indices = np.where((result_lall > 0) & (query_lall_vector > 0))[0].tolist()
                        hit_category_names = dataset_service.get_category_names_from_lall_indices(hit_category_indices)

                # 获取标签名称
                tag_names = dataset_service.get_tag_names_from_yall_indices(image_tags[:20])
                hit_tag_names = dataset_service.get_tag_names_from_yall_indices(hit_tags)

                # 统计命中
                if tag_hit:
                    total_tag_hits += 1
                if category_hit:
                    total_category_hits += 1

                # 获取结果哈希码
                result_hash_code = None
                if database_codes is not None and idx < len(database_codes):
                    result_hash_code = database_codes[idx].tolist()

                # 生成缩略图 URL
                config = DATASET_CONFIGS.get(request.dataset, {})
                if config.get('type') == 'raw':
                    # 使用静态文件 URL
                    original_id = dataset_service.get_original_image_id(image_id)
                    thumbnail_url = f"/flickr-images/im{original_id + 1}.jpg"
                else:
                    # 使用 API 端点
                    thumbnail_url = f"/api/images/{image_id}?format=image&dataset={request.dataset}"

                # 确保 distance 和 score 非负
                distance = max(0.0, float(distances_flat[idx]))
                score = float(1.0 / (1.0 + distance))

                results.append(SearchResult(
                    rank=rank + 1,
                    image_id=image_id,
                    score=score,
                    distance=distance,
                    tags=image_tags[:20] if image_tags else [],
                    tag_names=tag_names,
                    hit_tags=hit_tags,
                    hit_tag_names=hit_tag_names,
                    thumbnail_url=thumbnail_url,
                    hash_code=result_hash_code,
                    category_hit=category_hit,
                    tag_hit=tag_hit,
                    category_names=result_category_names,
                    hit_category_names=hit_category_names
                ))

            # 计算命中率统计（包含两种命中率）
            hit_stats = {
                "total_results": len(results),
                # 标签命中（YAll）
                "tag_hits": total_tag_hits,
                "tag_hit_rate": total_tag_hits / len(results) if results else 0,
                # 类别命中（LAll）- 与评估 mAP 对应
                "category_hits": total_category_hits,
                "category_hit_rate": total_category_hits / len(results) if results else 0,
                # 查询信息
                "query_tags": list(query_yall_indices),
                "query_tag_names": query_tag_names
            }

        search_time_ms = (time.time() - start_time) * 1000

        return SearchResponse(
            success=True,
            query_type=request.query_type,
            tag_indices=request.tag_indices,
            query_tag_names=query_tag_names if request.query_type != "image_to_tag" else [],
            results=results,
            tag_results=tag_results,
            total_results=len(results) + len(tag_results),
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
        image_codes, text_codes, tags = hash_cache.build_full_database_cache(
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
    tags = np.random.randint(0, 2, (num_images, 10)).astype(np.float32)

    # 加密
    aspe_service.encrypt_database(demo_hash_codes, tags)

    # 缓存数据库代码用于明文检索
    aspe_service.database_codes = demo_hash_codes

    logger.info(f"演示数据库构建完成 ({dataset})：{demo_hash_codes.shape}")


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
    tag_indices: str = "0,5,10",
    top_k: int = 5,
    dataset: str = Query(default="flickr25k", description="数据集名称")
):
    """
    演示搜索（简化版本）。

    参数：
    - tag_indices: 逗号分隔的标签索引，例如 "0,5,10"
    - top_k: 返回的结果数量
    - dataset: 数据集名称
    """
    try:
        # 解析标签索引
        indices = [int(x.strip()) for x in tag_indices.split(",") if x.strip().isdigit()]

        # 创建搜索请求
        request = SearchRequest(
            query_type="tag",
            tag_indices=indices,
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
        image_codes, text_codes, tags = hash_cache.build_full_database_cache(
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
