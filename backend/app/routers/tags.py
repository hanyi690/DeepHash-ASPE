"""
标签 API 路由

提供标签相关端点。
支持从原始数据文件或预定义列表获取标签名。
"""

from fastapi import APIRouter, Query

from app.services.dataset_service import get_dataset_service, DATASET_CONFIGS
from app.data.label_names import get_label_names, get_label_name, LABELS

router = APIRouter(prefix="/api/tags", tags=["tags"])


@router.get("")
async def get_labels(dataset: str = Query(default="flickr25k", description="数据集名称")):
    """
    返回数据集标签信息。

    参数：
        dataset: 数据集名称（默认 flickr25k）

    Flickr25K 使用 multi-hot 标签向量 (1386 维)，
    每个维度代表一个标签/类别。
    """
    try:
        dataset_service = get_dataset_service(dataset_name=dataset)
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
            "label_counts": label_counts[:100] if label_counts else [],
            "message": f"{dataset} 数据集包含 {total_images} 张图像，标签向量维度为 {label_dim}"
        }

    except Exception as e:
        return {
            "success": False,
            "total": 0,
            "label_dim": 1386,
            "message": f"获取标签信息失败：{str(e)}"
        }


@router.get("/stats")
async def get_label_stats(dataset: str = Query(default="flickr25k", description="数据集名称")):
    """
    返回标签统计信息。

    参数：
        dataset: 数据集名称（默认 flickr25k）

    包括每个标签的出现频率等。
    """
    try:
        dataset_service = get_dataset_service(dataset_name=dataset)
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


@router.get("/names/{dataset}")
async def get_dataset_label_names(dataset: str):
    """
    返回指定数据集的标签名称列表（YAll 索引顺序）。

    YAll 索引对应 .mat 文件中 YAll 矩阵的列顺序，
    用于跨模态检索时的标签向量构建。

    Args:
        dataset: 数据集名称 (flickr25k 或 nuswide)

    Returns:
        标签名称列表（YAll 索引顺序）
    """
    dataset_lower = dataset.lower()

    # 尝试从数据集服务获取 YAll 顺序的标签名
    try:
        dataset_service = get_dataset_service(dataset_name=dataset_lower)
        config = DATASET_CONFIGS.get(dataset_lower, {})

        if config.get('type') == 'raw':
            # 从 tag_mapping.npy 获取 YAll 索引对应的标签名
            from pathlib import Path
            import numpy as np

            tag_mapping_path = Path(config['data_path']) / 'tag_mapping.npy'
            tag_mapping_path_full = Path(__file__).parent.parent.parent.parent / tag_mapping_path

            if tag_mapping_path_full.exists():
                tag_mapping = np.load(str(tag_mapping_path_full), allow_pickle=True)
                return {
                    "success": True,
                    "dataset": dataset_lower,
                    "label_names": tag_mapping.tolist(),
                    "total": len(tag_mapping),
                    "source": "yall_mapping",
                    "message": f"获取 {dataset_lower} 标签名称成功（YAll 顺序），共 {len(tag_mapping)} 个标签"
                }

            # 回退到原始 common_tags 顺序
            tag_names = dataset_service.get_tag_names()
            if tag_names:
                return {
                    "success": True,
                    "dataset": dataset_lower,
                    "label_names": tag_names,
                    "total": len(tag_names),
                    "source": "common_tags",
                    "message": f"获取 {dataset_lower} 标签名称成功，共 {len(tag_names)} 个标签"
                }
    except Exception as e:
        # 回退到预定义列表
        pass

    # 使用预定义标签名
    label_names = get_label_names(dataset_lower)

    if not label_names:
        return {
            "success": False,
            "dataset": dataset,
            "label_names": [],
            "total": 0,
            "message": f"不支持的数据集: {dataset}，支持的数据集: {list(LABELS.keys())}"
        }

    return {
        "success": True,
        "dataset": dataset_lower,
        "label_names": label_names,
        "total": len(label_names),
        "source": "predefined",
        "message": f"获取 {dataset_lower} 标签名称成功，共 {len(label_names)} 个标签"
    }


@router.get("/names/{dataset}/{index}")
async def get_single_label_name(dataset: str, index: int):
    """
    返回指定数据集中某个索引的标签名称。

    Args:
        dataset: 数据集名称
        index: 标签索引

    Returns:
        标签名称
    """
    dataset_lower = dataset.lower()
    label_name = get_label_name(dataset_lower, index)

    label_names = get_label_names(dataset_lower)
    if not label_names:
        return {
            "success": False,
            "dataset": dataset,
            "index": index,
            "name": None,
            "message": f"不支持的数据集: {dataset}"
        }

    if index < 0 or index >= len(label_names):
        return {
            "success": False,
            "dataset": dataset_lower,
            "index": index,
            "name": None,
            "message": f"索引 {index} 超出范围 (0-{len(label_names)-1})"
        }

    return {
        "success": True,
        "dataset": dataset_lower,
        "index": index,
        "name": label_name,
        "message": "获取标签名称成功"
    }


@router.get("/datasets")
async def get_supported_datasets():
    """
    返回支持的数据集列表及其标签数量。
    """
    return {
        "success": True,
        "datasets": [
            {
                "name": "flickr25k",
                "label_count": len(LABELS.get("flickr25k", [])),
                "description": "Flickr25K 数据集"
            },
            {
                "name": "nuswide",
                "label_count": len(LABELS.get("nuswide", [])),
                "description": "NUS-WIDE 数据集"
            }
        ],
        "message": "获取支持的数据集列表成功"
    }