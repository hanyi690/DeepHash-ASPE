"""
重建哈希缓存脚本

使用已训练好的 DCMH 模型和 GPU 加速重建哈希缓存。
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np


def main():
    print("=" * 60)
    print("重建哈希缓存")
    print("=" * 60)

    # 检查 GPU
    if torch.cuda.is_available():
        print(f"\n✅ GPU 可用: {torch.cuda.get_device_name(0)}")
        use_gpu = True
    else:
        print("\n⚠️ GPU 不可用，使用 CPU")
        use_gpu = False

    # 导入服务
    from backend.app.services.dcmh_service import DCMHService
    from backend.app.services.dataset_service import DatasetService
    from backend.app.services.hash_cache_service import HashCacheService
    from backend.app.services.aspe_service import get_aspe_service

    # 指定模型路径
    model_dir = PROJECT_ROOT / "results" / "flickr-25k" / "20260324_090727"
    img_model_path = model_dir / "img_model.pth"
    txt_model_path = model_dir / "txt_model.pth"

    print(f"\n模型路径:")
    print(f"  - 图像模型: {img_model_path}")
    print(f"  - 文本模型: {txt_model_path}")
    print(f"  - 存在: {'✅' if img_model_path.exists() else '❌'}")

    # 初始化服务
    print("\n初始化 DCMH 服务...")
    dcmh_service = DCMHService(
        bit_dim=64,
        img_model_path=str(img_model_path),
        txt_model_path=str(txt_model_path),
        use_gpu=use_gpu
    )

    if dcmh_service.is_loaded():
        print("✅ 模型加载成功")
    else:
        print("❌ 模型加载失败")
        return

    # 初始化数据集服务
    print("\n加载数据集...")
    dataset_service = DatasetService()
    dataset_service.load_data()

    # 初始化缓存服务
    cache_service = HashCacheService(bit_dim=64)

    # 清除旧缓存
    print("\n清除旧缓存...")
    cache_service.clear_cache()

    # 构建数据库缓存
    print("\n构建数据库哈希码缓存...")
    print(f"  批次大小: 128")
    print(f"  使用 GPU: {use_gpu}")

    database_codes, database_labels = cache_service.build_database_cache(
        dcmh_service,
        dataset_service,
        batch_size=128,
        force_rebuild=True
    )

    print(f"\n✅ 数据库哈希码: {database_codes.shape}")
    print(f"✅ 数据库标签: {database_labels.shape}")

    # 构建查询集缓存
    print("\n构建查询集哈希码缓存...")
    query_codes, query_labels = cache_service.build_query_cache(
        dcmh_service,
        dataset_service,
        batch_size=128,
        force_rebuild=True
    )

    print(f"\n✅ 查询集哈希码: {query_codes.shape}")
    print(f"✅ 查询集标签: {query_labels.shape}")

    # 构建 ASPE 加密缓存
    print("\n构建 ASPE 加密缓存...")
    aspe_service = get_aspe_service()

    encrypted_database = cache_service.build_encrypted_cache(
        aspe_service,
        force_rebuild=True
    )

    print(f"\n✅ 加密数据库: {encrypted_database.shape}")

    # 验证缓存
    print("\n验证缓存...")
    cache_info = cache_service.get_cache_info()
    print(f"  缓存目录: {cache_info['cache_dir']}")
    print(f"  数据库缓存: {'✅' if cache_info['database_cached'] else '❌'}")
    print(f"  查询集缓存: {'✅' if cache_info['query_cached'] else '❌'}")
    print(f"  加密缓存: {'✅' if cache_info['encrypted_cached'] else '❌'}")
    print(f"  数据库大小: {cache_info['database_size']}")
    print(f"  查询集大小: {cache_info['query_size']}")

    print("\n" + "=" * 60)
    print("✅ 哈希缓存重建完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()