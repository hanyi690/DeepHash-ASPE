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
from app.services.dataset_service import get_dataset_service

router = APIRouter(prefix="/api/images", tags=["images"])


@router.post("/upload", response_model=ImageUploadResponse)
async def upload_image(file: UploadFile = File(...)):
    """上传图像文件。"""
    try:
        # 读取图像
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')

        # 生成临时 image_id
        image_id = hash(contents) % 1000000

        # 获取 DCMH 服务
        dcmh_service = get_dcmh_service()

        # 预处理图像
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        image_tensor = transform(image).unsqueeze(0)

        # 生成哈希码
        hash_code = dcmh_service.generate_image_code(image_tensor)

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
        dcmh_service = get_dcmh_service()
        dataset_service = get_dataset_service()

        # 获取图像
        if image_data:
            # 从 base64 解码
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            from torchvision import transforms
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])
            image_tensor = transform(image).unsqueeze(0)
        elif image_id:
            # 从数据集获取图像
            dataset_service.load_data()
            _, _, retrieval_indices = dataset_service.get_data_split_indices()

            # 查找索引
            if image_id in retrieval_indices:
                # 使用 DataLoader 加载图像
                dataloader = dataset_service.create_image_dataloader(
                    np.array([image_id]), batch_size=1
                )
                for batch in dataloader:
                    image_tensor = batch
                    break
            else:
                return ImageFeatureResponse(
                    success=False,
                    message=f"图像 {image_id} 不存在于检索库中"
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
        dataset_service = get_dataset_service()
        dataset_service.load_data()

        # 获取数据划分
        _, _, retrieval_indices = dataset_service.get_data_split_indices()

        # 检查图像是否在检索库中
        if image_id not in retrieval_indices:
            raise HTTPException(status_code=404, detail="Image not found in retrieval set")

        # 获取标签
        tags = dataset_service.get_tags(np.array([image_id]))
        labels = np.where(tags[0] > 0)[0].tolist() if tags is not None else []

        return {
            "success": True,
            "image_id": image_id,
            "labels": labels[:20],  # 返回前20个标签
            "total_labels": len(labels)
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/batch-hash")
async def batch_generate_hash_codes(image_ids: list[int]):
    """批量生成图像哈希码。"""
    try:
        dcmh_service = get_dcmh_service()
        dataset_service = get_dataset_service()
        dataset_service.load_data()

        # 获取检索库索引
        _, _, retrieval_indices = dataset_service.get_data_split_indices()
        retrieval_set = set(retrieval_indices.tolist())

        # 筛选有效 ID
        valid_ids = [idx for idx in image_ids if idx in retrieval_set]

        if not valid_ids:
            return {"success": False, "message": "没有有效的图像"}

        # 批量加载图像
        dataloader = dataset_service.create_image_dataloader(
            np.array(valid_ids), batch_size=len(valid_ids)
        )

        hash_codes_list = []
        for batch in dataloader:
            hash_codes = dcmh_service.generate_image_code(batch)
            hash_codes_list.append(hash_codes)

        if hash_codes_list:
            all_hash_codes = torch.cat(hash_codes_list, dim=0)
            return {
                "success": True,
                "hash_codes": all_hash_codes.tolist(),
                "image_ids": valid_ids,
                "num_images": len(valid_ids)
            }
        else:
            return {"success": False, "message": "哈希码生成失败"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
