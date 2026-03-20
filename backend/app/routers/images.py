"""
图像 API 路由

处理图像上传、特征提取和哈希码生成。
"""

from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import Optional
import torch
import numpy as np
from PIL import Image
import io
import base64

from app.schemas.search import ImageUploadResponse, ImageFeatureResponse
from app.services.dcmh_service import get_dcmh_service
from app.services.coco_service import get_coco_service

router = APIRouter(prefix="/api/images", tags=["images"])


@router.post("/upload", response_model=ImageUploadResponse)
async def upload_image(file: UploadFile = File(...)):
    """上传图像文件。"""
    try:
        # 读取图像
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')

        # 生成临时 image_id（实际应用中应该保存到数据库）
        image_id = hash(contents) % 1000000

        # 获取 COCO 服务
        coco_service = get_coco_service()

        # 预处理图像
        transform = coco_service.transform
        image_tensor = transform(image).unsqueeze(0)

        # 生成哈希码
        dcmh_service = get_dcmh_service()
        hash_code = dcmh_service.generate_image_code(image_tensor)

        # 存储图像（简化版本，实际应该保存）
        # 这里只是演示，返回 image_id 用于后续操作

        return ImageUploadResponse(
            success=True,
            image_id=image_id,
            message=f"成功上传图像，尺寸：{image.size}",
            image_url=f"/api/images/{image_id}"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/feature", response_model=ImageFeatureResponse)
async def extract_image_feature(
    image_id: Optional[int] = None,
    image_url: Optional[str] = None,
    image_data: Optional[str] = None
):
    """提取图像特征和生成哈希码。"""
    try:
        coco_service = get_coco_service()
        dcmh_service = get_dcmh_service()

        # 获取图像
        if image_data:
            # 从 base64 解码
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            transform = coco_service.transform
            image_tensor = transform(image).unsqueeze(0)
        elif image_id:
            # 从 COCO 服务获取
            image_tensor, metadata = coco_service.get_image(image_id)
            if image_tensor is None:
                return ImageFeatureResponse(
                    success=False,
                    message=f"图像 {image_id} 不存在"
                )
        else:
            return ImageFeatureResponse(
                success=False,
                message="必须提供 image_id 或 image_data"
            )

        # 生成哈希码
        hash_code = dcmh_service.generate_image_code(image_tensor)

        # 转换为列表
        hash_code_list = hash_code.squeeze().tolist()

        return ImageFeatureResponse(
            success=True,
            hash_code=hash_code_list,
            message="成功提取图像特征"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{image_id}")
async def get_image(image_id: int):
    """获取单个图像信息。"""
    try:
        coco_service = get_coco_service()

        # 获取图像
        image_tensor, metadata = coco_service.get_image(image_id)
        if image_tensor is None:
            raise HTTPException(status_code=404, detail="Image not found")

        # 获取标题
        captions = coco_service.get_captions(image_id)

        return {
            "success": True,
            "image_id": image_id,
            "metadata": metadata,
            "captions": captions
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/batch-hash")
async def batch_generate_hash_codes(image_ids: list[int]):
    """批量生成图像哈希码。"""
    try:
        coco_service = get_coco_service()
        dcmh_service = get_dcmh_service()

        # 收集图像张量
        images = []
        valid_ids = []

        for image_id in image_ids:
            image_tensor, _ = coco_service.get_image(image_id)
            if image_tensor is not None:
                images.append(image_tensor)
                valid_ids.append(image_id)

        if not images:
            return {"success": False, "message": "没有有效的图像"}

        # 批量生成哈希码
        images_tensor = torch.cat(images, dim=0)
        hash_codes = dcmh_service.generate_image_code(images_tensor)

        return {
            "success": True,
            "hash_codes": hash_codes.tolist(),
            "image_ids": valid_ids,
            "num_images": len(valid_ids)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
