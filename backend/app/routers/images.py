"""
图像 API 路由

处理图像上传、特征提取和哈希码生成。
"""

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import Response
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
async def get_image(image_id: int, format: str = "json"):
    """
    获取单个图像。

    参数：
        image_id: 图像索引
        format: 返回格式，'json' 返回元信息，'image' 返回图像数据
    """
    try:
        dataset_service = get_dataset_service()
        dataset_service.load_data()

        # 获取数据划分
        _, _, retrieval_indices = dataset_service.get_data_split_indices()

        # 查找图像在检索库中的位置
        idx_in_retrieval = None
        for i, idx in enumerate(retrieval_indices):
            if idx == image_id:
                idx_in_retrieval = i
                break

        if idx_in_retrieval is None:
            raise HTTPException(status_code=404, detail="Image not found in retrieval set")

        # 获取标签
        tags = dataset_service.get_tags(np.array([image_id]))
        labels = np.where(tags[0] > 0)[0].tolist() if tags is not None else []

        if format == "image":
            # 返回实际图像数据
            import h5py
            h5_file = h5py.File(dataset_service.data_path, 'r')
            try:
                # 图像数据存储在 h5 文件中
                images = h5_file['images']
                if image_id >= len(images):
                    raise HTTPException(status_code=404, detail="Image index out of range")

                # 获取图像数据
                # h5 中的图像是 int8 数组 [3, H, W]，范围约 [-42, 127]
                img_data = images[image_id]

                # CHW -> HWC
                img_data = np.transpose(img_data, (1, 2, 0))

                # 转换为 uint8
                # 原始数据范围约 [-42, 127]，需要映射到 [0, 255]
                img_data = img_data.astype(np.float32)
                img_data = (img_data - img_data.min()) / (img_data.max() - img_data.min() + 1e-8) * 255
                img_data = img_data.astype(np.uint8)

                # 创建 PIL 图像
                pil_image = Image.fromarray(img_data)

                # 转换为 PNG
                img_buffer = io.BytesIO()
                pil_image.save(img_buffer, format='PNG')
                img_buffer.seek(0)

                return Response(
                    content=img_buffer.getvalue(),
                    media_type="image/png"
                )
            finally:
                h5_file.close()

        # 默认返回 JSON 元信息
        return {
            "success": True,
            "image_id": image_id,
            "labels": labels[:20],  # 返回前20个标签
            "total_labels": len(labels),
            "thumbnail_url": f"/api/images/{image_id}?format=image"
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
