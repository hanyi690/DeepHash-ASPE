"""
搜索 API 路由

提供隐私保护跨模态检索端点。
"""

from fastapi import APIRouter, HTTPException
import numpy as np
import time
from typing import List, Optional
import torch

from app.schemas.search import (
    SearchRequest,
    SearchResponse,
    SearchResult,
    EncryptionInfo,
    ImageToLabelResult
)
from app.services.aspe_service import get_aspe_service
from app.services.dcmh_service import get_dcmh_service
from app.services.dataset_service import get_dataset_service
from app.services.hash_cache_service import get_hash_cache_service

router = APIRouter(prefix="/api/search", tags=["search"])

# 缓存初始化标志
_cache_initialized = False


async def ensure_cache_initialized():
    """确保缓存已初始化。"""
    global _cache_initialized

    if _cache_initialized:
        return

    try:
        hash_cache = get_hash_cache_service()
        aspe_service = get_aspe_service()
        dcmh_service = get_dcmh_service()
        dataset_service = get_dataset_service()

        # 检查模型是否已加载
        if not dcmh_service.is_loaded():
            print("警告：DCMH 模型未加载，将使用随机哈希码")

        # 检查缓存是否存在
        if hash_cache.exists("database"):
            # 加载缓存
            database_codes, database_labels = hash_cache.load_cache("database")
            if database_codes is not None:
                hash_cache.database_codes = database_codes
                hash_cache.database_labels = database_labels

                # 加密数据库
                if hash_cache.exists("encrypted_database"):
                    encrypted = hash_cache.load_encrypted_cache()
                    if encrypted is not None:
                        hash_cache.encrypted_database = encrypted
                        aspe_service.encrypted_database = encrypted
                        aspe_service.database_codes = database_codes
                        aspe_service.database_labels = database_labels
                else:
                    # 加密数据库
                    encrypted = aspe_service.encrypt_database(database_codes, database_labels)
                    hash_cache.encrypted_database = encrypted

                aspe_service.database_codes = database_codes
                aspe_service.database_labels = database_labels
                print(f"缓存已加载：{database_codes.shape[0]} 条记录")

        _cache_initialized = True

    except Exception as e:
        print(f"缓存初始化警告：{e}")
        _cache_initialized = True


@router.post("", response_model=SearchResponse)
async def search(request: SearchRequest):
    """
    执行跨模态检索。

    支持：
    - label_to_image: 标签→图像检索
    - image_to_label: 图像→标签检索
    - image_to_image: 图像→图像检索
    - 加密/明文检索
    """
    try:
        start_time = time.time()

        # 确保缓存初始化
        await ensure_cache_initialized()

        aspe_service = get_aspe_service()
        dcmh_service = get_dcmh_service()
        hash_cache = get_hash_cache_service()
        dataset_service = get_dataset_service()

        # 检查数据库是否已构建
        if aspe_service.encrypted_database is None and hash_cache.database_codes is None:
            # 如果没有缓存，尝试构建
            if dcmh_service.is_loaded():
                await _build_database_from_real_data()
            else:
                # 回退到演示模式
                await _build_demo_database()

        # 生成查询哈希码
        query_code_np = None
        is_text_query = False  # 标记是否为文本查询

        if request.query_type == "label_to_image" or request.query_type == "label":
            # 标签→图像检索
            if not request.label_indices:
                return SearchResponse(
                    success=False,
                    query_type="label_to_image",
                    label_indices=[],
                    results=[],
                    total_results=0,
                    search_time_ms=0
                )

            # 构建 multi-hot 标签向量 (1386 维)
            dataset_service.load_data()
            tags = dataset_service.get_tags()
            label_dim = tags.shape[1] if tags is not None else 1386

            label_vector = np.zeros(label_dim, dtype=np.float32)
            for idx in request.label_indices:
                if 0 <= idx < label_dim:
                    label_vector[idx] = 1.0

            # 文本模态 → 哈希码
            text_tensor = torch.from_numpy(label_vector).unsqueeze(0).float()
            query_code = dcmh_service.generate_text_code_single(text_tensor)
            query_code_np = query_code.cpu().numpy().squeeze()
            is_text_query = True

        elif request.query_type in ["image_to_label", "image_to_image"]:
            # 图像→标签/图像检索
            if request.query_image:
                # 处理 base64 图像
                query_code_np = _process_base64_image(
                    request.query_image, dcmh_service, dataset_service
                )
            else:
                query_code_np = _generate_demo_query_code(dcmh_service.bit_dim).squeeze()
            is_text_query = False

        else:
            return SearchResponse(
                success=False,
                query_type=request.query_type,
                label_indices=request.label_indices,
                results=[],
                total_results=0,
                search_time_ms=0
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
            database_codes = hash_cache.database_codes or aspe_service.database_codes
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
        retrieval_labels = dataset_service.get_tags(retrieval_indices)

        # 根据查询类型构建结果
        results: List[SearchResult] = []
        label_results: List[ImageToLabelResult] = []

        if request.query_type == "image_to_label":
            # 图像→标签检索：返回相似图像的标签
            for rank, idx in enumerate(top_k_indices):
                # 获取实际图像索引
                actual_idx = retrieval_indices[idx] if idx < len(retrieval_indices) else idx
                image_id = int(actual_idx)

                # 获取该图像的标签索引列表
                image_labels = []
                if retrieval_labels is not None and idx < len(retrieval_labels):
                    label_row = retrieval_labels[idx]
                    image_labels = np.where(label_row > 0)[0].tolist()

                label_results.append(ImageToLabelResult(
                    rank=rank + 1,
                    image_id=image_id,
                    labels=image_labels[:20] if image_labels else [],  # 最多返回20个标签
                    score=float(1.0 / (1.0 + distances_flat[idx])),
                    distance=float(distances_flat[idx])
                ))
        else:
            # 标签→图像 / 图像→图像检索：返回图像结果
            for rank, idx in enumerate(top_k_indices):
                # 获取实际图像索引
                actual_idx = retrieval_indices[idx] if idx < len(retrieval_indices) else idx
                image_id = int(actual_idx)

                # 获取该图像的标签索引列表
                image_labels = []
                if retrieval_labels is not None and idx < len(retrieval_labels):
                    label_row = retrieval_labels[idx]
                    image_labels = np.where(label_row > 0)[0].tolist()

                # 获取结果哈希码
                result_hash_code = None
                if hash_cache.database_codes is not None and idx < len(hash_cache.database_codes):
                    result_hash_code = hash_cache.database_codes[idx].tolist()

                results.append(SearchResult(
                    rank=rank + 1,
                    image_id=image_id,
                    score=float(1.0 / (1.0 + distances_flat[idx])),
                    distance=float(distances_flat[idx]),
                    labels=image_labels[:20] if image_labels else [],  # 最多返回20个标签
                    thumbnail_url=f"/api/images/{image_id}",
                    hash_code=result_hash_code
                ))

        search_time_ms = (time.time() - start_time) * 1000

        return SearchResponse(
            success=True,
            query_type=request.query_type,
            label_indices=request.label_indices,
            results=results,
            label_results=label_results,
            total_results=len(results) + len(label_results),
            search_time_ms=search_time_ms,
            encryption_info=encryption_info,
            query_hash_code=query_code_np.tolist() if query_code_np is not None else None
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


async def _build_database_from_real_data():
    """使用真实数据构建数据库。"""
    hash_cache = get_hash_cache_service()
    dcmh_service = get_dcmh_service()
    dataset_service = get_dataset_service()
    aspe_service = get_aspe_service()

    try:
        # 构建数据库缓存
        database_codes, database_labels = hash_cache.build_database_cache(
            dcmh_service, dataset_service, batch_size=32, force_rebuild=False
        )

        # 构建加密缓存
        hash_cache.build_encrypted_cache(aspe_service, force_rebuild=False)

        print(f"真实数据库构建完成：{database_codes.shape}")

    except Exception as e:
        print(f"构建真实数据库失败：{e}")
        await _build_demo_database()


async def _build_demo_database():
    """构建演示数据库。"""
    aspe_service = get_aspe_service()
    dcmh_service = get_dcmh_service()

    # 构建演示数据库
    num_images = 100
    demo_hash_codes = np.sign(np.random.randn(num_images, dcmh_service.bit_dim))

    # 生成随机标签
    labels = np.random.randint(0, 2, (num_images, 10)).astype(np.float32)

    # 加密
    aspe_service.encrypt_database(demo_hash_codes, labels)

    # 缓存数据库代码用于明文检索
    aspe_service.database_codes = demo_hash_codes

    print(f"演示数据库构建完成：{demo_hash_codes.shape}")


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
    from app.services.dcmh_service import preprocess_image

    # 解码 base64
    if ',' in base64_str:
        base64_str = base64_str.split(',')[1]

    image_data = base64.b64decode(base64_str)
    image = Image.open(BytesIO(image_data))

    # 预处理图像
    image_tensor = preprocess_image(image).unsqueeze(0)

    # 生成哈希码
    hash_code = dcmh_service.generate_image_code_single(image_tensor)
    return hash_code.cpu().numpy().squeeze()


@router.get("/demo")
async def demo_search(label_indices: str = "0,5,10", top_k: int = 5):
    """
    演示搜索（简化版本）。

    参数：
    - label_indices: 逗号分隔的标签索引，例如 "0,5,10"
    - top_k: 返回的结果数量
    """
    try:
        # 解析标签索引
        indices = [int(x.strip()) for x in label_indices.split(",") if x.strip().isdigit()]

        # 创建搜索请求
        request = SearchRequest(
            query_type="label",
            label_indices=indices,
            top_k=top_k,
            use_encrypted=True
        )

        # 执行搜索
        return await search(request)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status")
async def search_status():
    """获取搜索服务状态。"""
    hash_cache = get_hash_cache_service()
    dcmh_service = get_dcmh_service()
    aspe_service = get_aspe_service()

    return {
        "cache_info": hash_cache.get_cache_info(),
        "dcmh_status": dcmh_service.get_status(),
        "aspe_status": aspe_service.get_status(),
        "cache_initialized": _cache_initialized
    }


@router.post("/rebuild-cache")
async def rebuild_cache():
    """重建哈希码缓存。"""
    global _cache_initialized
    _cache_initialized = False

    try:
        hash_cache = get_hash_cache_service()
        dcmh_service = get_dcmh_service()
        dataset_service = get_dataset_service()
        aspe_service = get_aspe_service()

        if not dcmh_service.is_loaded():
            return {"success": False, "message": "DCMH 模型未加载"}

        # 重建缓存
        database_codes, database_labels = hash_cache.build_database_cache(
            dcmh_service, dataset_service, batch_size=32, force_rebuild=True
        )

        hash_cache.build_encrypted_cache(aspe_service, force_rebuild=True)

        _cache_initialized = True

        return {
            "success": True,
            "message": "缓存重建完成",
            "database_size": database_codes.shape[0]
        }

    except Exception as e:
        return {"success": False, "message": str(e)}