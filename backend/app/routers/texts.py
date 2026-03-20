"""
文本 API 路由

处理文本输入、特征提取和哈希码生成。
"""

from fastapi import APIRouter, HTTPException
from typing import Optional
import torch
import numpy as np

from app.schemas.search import TextProcessRequest, TextProcessResponse, HashCodeResponse
from app.services.dcmh_service import get_dcmh_service
from app.services.coco_service import get_coco_service

router = APIRouter(prefix="/api/texts", tags=["texts"])


@router.post("/process", response_model=TextProcessResponse)
async def process_text(request: TextProcessRequest):
    """处理文本并生成哈希码。"""
    try:
        coco_service = get_coco_service()
        dcmh_service = get_dcmh_service()

        # 将文本转换为向量
        text_vector = coco_service.text_to_vector(request.text)
        text_tensor = text_vector.unsqueeze(0)  # [1, 2000]

        # 生成哈希码
        hash_code = dcmh_service.generate_text_code(text_tensor)

        # 转换为列表
        hash_code_list = hash_code.squeeze().tolist()

        return TextProcessResponse(
            success=True,
            text=request.text,
            hash_code=hash_code_list,
            message="成功处理文本并生成哈希码"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/hash", response_model=HashCodeResponse)
async def generate_text_hash(request: TextProcessRequest):
    """从文本生成哈希码。"""
    try:
        coco_service = get_coco_service()
        dcmh_service = get_dcmh_service()

        # 将文本转换为向量
        text_vector = coco_service.text_to_vector(request.text)
        text_tensor = text_vector.unsqueeze(0)

        # 生成哈希码
        hash_code = dcmh_service.generate_text_code(text_tensor)

        hash_code_list = hash_code.squeeze().tolist()

        return HashCodeResponse(
            success=True,
            hash_code=hash_code_list,
            bit_dim=dcmh_service.bit_dim,
            message="成功生成文本哈希码"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/batch-hash")
async def batch_generate_text_hash(texts: list[str]):
    """批量生成文本哈希码。"""
    try:
        coco_service = get_coco_service()
        dcmh_service = get_dcmh_service()

        # 处理所有文本
        text_tensors = []
        for text in texts:
            text_vector = coco_service.text_to_vector(text)
            text_tensors.append(text_vector)

        # 批量生成哈希码
        texts_tensor = torch.stack(text_tensors)
        hash_codes = dcmh_service.generate_text_code(texts_tensor)

        return {
            "success": True,
            "hash_codes": hash_codes.tolist(),
            "num_texts": len(texts),
            "bit_dim": dcmh_service.bit_dim
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
