"""
哈希码 API 路由

提供哈希码生成和管理端点。
"""

from fastapi import APIRouter, HTTPException
from typing import Optional, List
import torch
import numpy as np

from app.schemas.search import HashCodeRequest, HashCodeResponse
from app.services.dcmh_service import get_dcmh_service
from app.services.dataset_service import get_dataset_service

router = APIRouter(prefix="/api/hash", tags=["hash"])


@router.post("/image", response_model=HashCodeResponse)
async def generate_image_hash(image_data: dict):
    """从图像数据生成哈希码。"""
    try:
        dcmh_service = get_dcmh_service()
        dataset_service = get_dataset_service()
        dataset_service.load_data()

        # 解析输入
        if "image_id" in image_data:
            image_id = image_data["image_id"]

            # 检查图像是否在检索库中
            _, _, retrieval_indices = dataset_service.get_data_split_indices()
            if image_id not in retrieval_indices:
                return HashCodeResponse(
                    success=False,
                    message=f"图像 {image_id} 不在检索库中"
                )

            # 加载图像
            dataloader = dataset_service.create_image_dataloader(
                np.array([image_id]), batch_size=1
            )
            for batch in dataloader:
                image_tensor = batch
                break

        elif "tensor" in image_data:
            # 直接使用张量数据
            tensor_data = image_data["tensor"]
            image_tensor = torch.tensor(tensor_data)
            if image_tensor.dim() == 3:
                image_tensor = image_tensor.unsqueeze(0)
        else:
            return HashCodeResponse(
                success=False,
                message="必须提供 image_id 或 tensor"
            )

        # 生成哈希码
        hash_code = dcmh_service.generate_image_code(image_tensor)
        hash_code_list = hash_code.squeeze().tolist()

        return HashCodeResponse(
            success=True,
            hash_code=hash_code_list,
            bit_dim=dcmh_service.bit_dim,
            message="成功生成图像哈希码"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/text", response_model=HashCodeResponse)
async def generate_text_hash(text_data: dict):
    """从标签向量生成哈希码。"""
    try:
        dcmh_service = get_dcmh_service()
        dataset_service = get_dataset_service()
        dataset_service.load_data()

        # 获取标签维度
        tags = dataset_service.get_tags()
        tag_dim = tags.shape[1] if tags is not None else 1386

        # 解析输入
        if "tag_indices" in text_data:
            # 从标签索引构建向量
            tag_indices = text_data["tag_indices"]
            tag_vector = np.zeros(tag_dim, dtype=np.float32)
            for idx in tag_indices:
                if 0 <= idx < tag_dim:
                    tag_vector[idx] = 1.0
            text_tensor = torch.from_numpy(tag_vector).unsqueeze(0).float()
        elif "vector" in text_data:
            vector_data = text_data["vector"]
            text_tensor = torch.tensor(vector_data).unsqueeze(0).float()
        else:
            return HashCodeResponse(
                success=False,
                message="必须提供 tag_indices 或 vector"
            )

        # 生成哈希码
        hash_code = dcmh_service.generate_text_code_single(text_tensor)
        hash_code_list = hash_code.squeeze().tolist()

        return HashCodeResponse(
            success=True,
            hash_code=hash_code_list,
            bit_dim=dcmh_service.bit_dim,
            message="成功生成文本哈希码"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status")
async def get_hash_status():
    """获取哈希服务状态。"""
    try:
        dcmh_service = get_dcmh_service()
        return {
            "success": True,
            "bit_dim": dcmh_service.bit_dim,
            "use_gpu": dcmh_service.use_gpu,
            "models_loaded": True
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
