"""
文本/标签 API 路由

处理标签向量输入和哈希码生成。
"""

from fastapi import APIRouter, HTTPException
from typing import Optional, List
from pydantic import BaseModel, Field
import torch
import numpy as np

from app.schemas.search import HashCodeResponse
from app.services.dcmh_service import get_dcmh_service
from app.services.dataset_service import get_dataset_service

router = APIRouter(prefix="/api/texts", tags=["texts"])


class TagVectorRequest(BaseModel):
    """标签向量请求。"""
    tag_indices: List[int] = Field(..., description="标签索引列表")


@router.post("/hash", response_model=HashCodeResponse)
async def generate_tag_hash(tag_indices: List[int]):
    """
    从标签索引生成哈希码。

    参数：
    - tag_indices: 标签索引列表，例如 [0, 5, 10]

    返回：
    - 哈希码
    """
    try:
        dcmh_service = get_dcmh_service()
        dataset_service = get_dataset_service()
        dataset_service.load_data()

        # 获取标签维度
        tags = dataset_service.get_tags()
        tag_dim = tags.shape[1] if tags is not None else 1386

        # 构建 multi-hot 向量
        tag_vector = np.zeros(tag_dim, dtype=np.float32)
        for idx in tag_indices:
            if 0 <= idx < tag_dim:
                tag_vector[idx] = 1.0

        # 转换为张量
        text_tensor = torch.from_numpy(tag_vector).unsqueeze(0).float()

        # 生成哈希码
        hash_code = dcmh_service.generate_text_code_single(text_tensor)
        hash_code_list = hash_code.squeeze().tolist()

        return HashCodeResponse(
            success=True,
            hash_code=hash_code_list,
            bit_dim=dcmh_service.bit_dim,
            message=f"成功从 {len(tag_indices)} 个标签生成哈希码"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/batch-hash")
async def batch_generate_tag_hash(tag_indices_list: List[List[int]]):
    """
    批量从标签索引生成哈希码。

    参数：
    - tag_indices_list: 标签索引列表的列表

    返回：
    - 哈希码列表
    """
    try:
        dcmh_service = get_dcmh_service()
        dataset_service = get_dataset_service()
        dataset_service.load_data()

        # 获取标签维度
        tags = dataset_service.get_tags()
        tag_dim = tags.shape[1] if tags is not None else 1386

        # 构建标签向量矩阵
        tag_vectors = []
        for tag_indices in tag_indices_list:
            vec = np.zeros(tag_dim, dtype=np.float32)
            for idx in tag_indices:
                if 0 <= idx < tag_dim:
                    vec[idx] = 1.0
            tag_vectors.append(vec)

        # 转换为张量
        texts_tensor = torch.from_numpy(np.stack(tag_vectors)).float()

        # 批量生成哈希码
        hash_codes = dcmh_service.generate_text_code(texts_tensor)

        return {
            "success": True,
            "hash_codes": hash_codes.tolist(),
            "num_texts": len(tag_indices_list),
            "bit_dim": dcmh_service.bit_dim
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/dimension")
async def get_tag_dimension():
    """获取标签向量维度。"""
    try:
        dataset_service = get_dataset_service()
        dataset_service.load_data()

        tags = dataset_service.get_tags()
        tag_dim = tags.shape[1] if tags is not None else 1386

        return {
            "success": True,
            "tag_dim": tag_dim,
            "message": f"标签向量维度为 {tag_dim}"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))