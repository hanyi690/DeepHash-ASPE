"""
标签 API 路由

提供 Flickr25K 标签相关端点。
"""

from fastapi import APIRouter

from app.services.dataset_service import get_dataset_service

router = APIRouter(prefix="/api/labels", tags=["labels"])


@router.get("")
async def get_labels():
    """
    返回 Flickr25K 标签信息。

    Flickr25K 使用 multi-hot 标签向量 (1386 维)，
    每个维度代表一个标签/类别。
    """
    try:
        dataset_service = get_dataset_service()
        dataset_service.load_data()

        # 获取标签维度
        tags = dataset_service.get_tags()
        label_dim = tags.shape[1] if tags is not None else 1386

        # 获取总图像数
        total_images = dataset_service.get_image_count()

        # 获取标签统计信息
        label_counts = tags.sum(axis=0).tolist() if tags is not None else []

        return {
            "success": True,
            "total": total_images,
            "label_dim": label_dim,
            "label_counts": label_counts[:100] if label_counts else [],  # 返回前100个标签的计数
            "message": f"Flickr25K 数据集包含 {total_images} 张图像，标签向量维度为 {label_dim}"
        }

    except Exception as e:
        return {
            "success": False,
            "total": 0,
            "label_dim": 1386,
            "message": f"获取标签信息失败：{str(e)}"
        }


@router.get("/stats")
async def get_label_stats():
    """
    返回标签统计信息。

    包括每个标签的出现频率等。
    """
    try:
        dataset_service = get_dataset_service()
        dataset_service.load_data()

        # 获取所有标签
        tags = dataset_service.get_tags()

        if tags is None:
            return {
                "success": False,
                "message": "无法加载标签数据"
            }

        # 计算统计信息
        label_counts = tags.sum(axis=0)
        total_samples = tags.shape[0]

        # 找出最常见的标签
        top_indices = label_counts.argsort()[-20:][::-1].tolist()
        top_counts = label_counts[top_indices].tolist()

        # 标签密度
        avg_labels_per_sample = tags.sum() / total_samples

        return {
            "success": True,
            "total_samples": total_samples,
            "label_dim": tags.shape[1],
            "avg_labels_per_sample": float(avg_labels_per_sample),
            "top_labels": [
                {"index": idx, "count": int(count)}
                for idx, count in zip(top_indices, top_counts)
            ],
            "message": "标签统计信息获取成功"
        }

    except Exception as e:
        return {
            "success": False,
            "message": f"获取标签统计失败：{str(e)}"
        }