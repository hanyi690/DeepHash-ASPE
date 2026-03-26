"""
图像 API 路由

处理图像上传、特征提取和哈希码生成。
支持从原始 JPG 文件或 .mat 文件加载图像。
"""

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import Response
from typing import Optional
import torch
import numpy as np
from PIL import Image
import io
import base64
from pathlib import Path

from app.schemas.search import ImageUploadResponse, ImageFeatureResponse
from app.services.dcmh_service import get_dcmh_service
from app.services.dataset_service import get_dataset_service, DATASET_CONFIGS

# VGG-F ImageNet 均值（BGR 格式，用于恢复 .mat 中的预处理图像）
# B=123.66, G=116.77, R=103.93
VGGF_MEAN_BGR = np.array([123.66, 116.77, 103.93], dtype=np.float32).reshape(1, 1, 3)

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
async def get_image(
    image_id: int,
    format: str = "json",
    dataset: str = "flickr25k"
):
    """
    获取单个图像。

    参数：
        image_id: 图像索引（从 0 开始，对应 im{image_id+1}.jpg）
        format: 返回格式，'json' 返回元信息，'image' 返回图像数据
        dataset: 数据集名称
    """
    try:
        dataset_service = get_dataset_service(dataset_name=dataset)
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
        tags_data = dataset_service.get_tags(np.array([image_id]))
        tags = np.where(tags_data[0] > 0)[0].tolist() if tags_data is not None else []

        if format == "image":
            # 检查数据集类型
            config = DATASET_CONFIGS.get(dataset, {})
            if config.get('type') == 'raw':
                # 从原始 JPG 文件加载
                return _load_raw_image(image_id, dataset_service)
            else:
                # 从 .mat 文件加载
                return _load_mat_image(image_id, dataset_service)

        # 默认返回 JSON 元信息
        config = DATASET_CONFIGS.get(dataset, {})
        if config.get('type') == 'raw':
            original_id = dataset_service.get_original_image_id(image_id)
            thumbnail_url = f"/flickr-images/im{original_id + 1}.jpg"
        else:
            thumbnail_url = f"/api/images/{image_id}?format=image&dataset={dataset}"

        return {
            "success": True,
            "image_id": image_id,
            "tags": tags[:20],
            "total_tags": len(tags),
            "thumbnail_url": thumbnail_url
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _load_raw_image(image_id: int, dataset_service) -> Response:
    """
    从原始 JPG 文件加载图像。

    参数：
        image_id: 图像索引（从 0 开始）
        dataset_service: 数据集服务实例

    返回：
        FastAPI Response 对象
    """
    # 获取图像路径
    img_path = dataset_service.get_image_path(image_id)

    if not img_path.exists():
        raise HTTPException(status_code=404, detail=f"Image file not found: {img_path}")

    # 直接读取并返回图像
    with open(img_path, 'rb') as f:
        img_data = f.read()

    # 检测图像格式
    img = Image.open(io.BytesIO(img_data))
    format_map = {
        'JPEG': 'image/jpeg',
        'PNG': 'image/png',
        'GIF': 'image/gif',
        'BMP': 'image/bmp'
    }
    media_type = format_map.get(img.format, 'image/jpeg')

    return Response(content=img_data, media_type=media_type)


def _load_mat_image(image_id: int, dataset_service) -> Response:
    """
    从 .mat 文件加载并恢复图像。

    参数：
        image_id: 图像索引（从 0 开始）
        dataset_service: 数据集服务实例

    返回：
        FastAPI Response 对象
    """
    import h5py

    h5_file = h5py.File(dataset_service.mat_path, 'r')
    try:
        images = h5_file['images']
        if image_id >= len(images):
            raise HTTPException(status_code=404, detail="Image index out of range")

        # 获取图像数据 (CHW, int8)
        img_data = images[image_id]

        # 恢复图像
        img_data = img_data.astype(np.float32)  # CHW
        # 减均值后的数据 + 均值 = 原始数据（近似）
        img_data = img_data + VGGF_MEAN_BGR.reshape(3, 1, 1)
        img_data = np.clip(img_data, 0, 255).astype(np.uint8)

        # CHW -> HWC
        img_data = img_data.transpose(1, 2, 0)

        # BGR -> RGB
        img_data = img_data[:, :, ::-1]

        # 修正方向：顺时针旋转 90 度 + 左右翻转
        # .mat 中图像存储时经过了 MATLAB 的转置操作
        img_data = np.rot90(img_data, k=3)  # 顺时针 90 度
        img_data = np.fliplr(img_data)  # 左右翻转

        # 创建 PIL 图像
        pil_image = Image.fromarray(img_data)

        # 转换为 JPEG
        img_buffer = io.BytesIO()
        pil_image.save(img_buffer, format='JPEG', quality=85)
        img_buffer.seek(0)

        return Response(
            content=img_buffer.getvalue(),
            media_type="image/jpeg"
        )
    finally:
        h5_file.close()


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