"""
搜索 API 路由

提供隐私保护跨模态检索端点。
"""

from fastapi import APIRouter, HTTPException
import numpy as np
import time
from typing import List

from app.schemas.search import (
    SearchRequest,
    SearchResponse,
    SearchResult
)
from app.services.aspe_service import get_aspe_service
from app.services.dcmh_service import get_dcmh_service
from app.services.coco_service import get_coco_service

router = APIRouter(prefix="/api/search", tags=["search"])


@router.post("", response_model=SearchResponse)
async def search(request: SearchRequest):
    """
    执行跨模态检索。

    支持：
    - 文本→图像检索
    - 图像→图像检索（相同模态）
    - 加密/明文检索
    """
    try:
        start_time = time.time()

        aspe_service = get_aspe_service()
        dcmh_service = get_dcmh_service()
        coco_service = get_coco_service()

        # 检查数据库是否已构建
        if aspe_service.encrypted_database is None:
            # 自动构建数据库
            await _build_database_if_needed()

        # 生成查询哈希码
        if request.query_type == "text":
            if not request.query_text:
                return SearchResponse(
                    success=False,
                    query_type="text",
                    query_text="",
                    results=[],
                    total_results=0,
                    search_time_ms=0,
                    message="文本查询必须提供 query_text"
                )

            # 文本→哈希码
            text_vector = coco_service.text_to_vector(request.query_text)
            query_code = dcmh_service.generate_text_code(text_vector.unsqueeze(0))

        elif request.query_type == "image":
            # 图像→哈希码（简化处理）
            # 实际应该处理上传的图像
            query_code = _generate_demo_query_code(dcmh_service.bit_dim)

        else:
            return SearchResponse(
                success=False,
                query_type=request.query_type,
                results=[],
                total_results=0,
                search_time_ms=0,
                message=f"不支持的查询类型：{request.query_type}"
            )

        # 转换为 numpy
        query_code_np = query_code.cpu().numpy()

        # 执行检索
        if request.use_encrypted:
            # ASPE 加密检索
            encrypted_query = aspe_service.generate_trapdoor(query_code_np)
            distances = aspe_service.compute_ciphertext_distances(encrypted_query)

            # 计算密文 mAP（如果可能）
            plaintext_map = None
            ciphertext_map = None
            map_difference = None

        else:
            # 明文检索
            database_codes = aspe_service.database_codes
            if database_codes is None:
                return SearchResponse(
                    success=False,
                    query_type=request.query_type,
                    results=[],
                    total_results=0,
                    search_time_ms=0,
                    message="检索数据库未初始化"
                )

            distances = aspe_service._plaintext_hamming_distance(
                query_code_np, database_codes
            )

            plaintext_map = None
            ciphertext_map = None
            map_difference = None

        # 获取Top-K结果
        distances_flat = distances.squeeze()
        top_k_indices = np.argsort(distances_flat)[:request.top_k]

        # 构建结果
        results: List[SearchResult] = []
        image_ids = list(range(len(top_k_indices)))  # 简化 ID 映射

        for rank, idx in enumerate(top_k_indices):
            image_id = int(idx)
            captions = coco_service.get_captions(image_id)

            results.append(SearchResult(
                rank=rank + 1,
                image_id=image_id,
                score=float(1.0 / (1.0 + distances_flat[idx])),  # 相似度分数
                distance=float(distances_flat[idx]),
                captions=captions[:3] if captions else [],  # 最多显示 3 个标题
                thumbnail_url=f"/api/images/{image_id}"
            ))

        search_time_ms = (time.time() - start_time) * 1000

        return SearchResponse(
            success=True,
            query_type=request.query_type,
            query_text=request.query_text if request.query_type == "text" else None,
            results=results,
            total_results=len(results),
            search_time_ms=search_time_ms,
            plaintext_map=plaintext_map,
            ciphertext_map=ciphertext_map,
            map_difference=map_difference
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def _build_database_if_needed():
    """如果数据库未构建，则自动构建。"""
    aspe_service = get_aspe_service()
    dcmh_service = get_dcmh_service()
    coco_service = get_coco_service()

    if aspe_service.encrypted_database is not None:
        return

    # 构建演示数据库
    num_images = 100
    demo_hash_codes = np.sign(np.random.randn(num_images, dcmh_service.bit_dim))

    # 生成随机标签
    labels = np.random.randint(0, 2, (num_images, 10)).astype(np.float32)

    # 加密
    aspe_service.encrypt_database(demo_hash_codes, labels)

    # 缓存数据库代码用于明文检索
    aspe_service.database_codes = demo_hash_codes


def _generate_demo_query_code(bit_dim: int) -> np.ndarray:
    """生成演示查询哈希码。"""
    return np.sign(np.random.randn(1, bit_dim))


@router.get("/demo")
async def demo_search(query_text: str = "A cat sitting on a chair", top_k: int = 5):
    """演示搜索（简化版本）。"""
    try:
        # 创建搜索请求
        request = SearchRequest(
            query_type="text",
            query_text=query_text,
            top_k=top_k,
            use_encrypted=True
        )

        # 执行搜索
        return await search(request)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
