#!/usr/bin/env python3
"""
统一缓存构建脚本

支持构建 DCMH 和 CIR 的所有数据集缓存
- DCMH: flickr25k, nuswide（包含图像哈希码、文本哈希码）
- CIR: roxford5k, rparis6k（包含明文特征、加密特征）

关键改进：
- DCMH 同时构建图像和文本哈希码
- 密钥持久化

使用方法:
    # 构建所有缓存
    python scripts/build_all_cache.py --type all

    # 仅构建 DCMH 缓存
    python scripts/build_all_cache.py --type dcmh

    # 仅构建 CIR 缓存
    python scripts/build_all_cache.py --type cir

    # 指定数据集
    python scripts/build_all_cache.py --type dcmh --dataset flickr25k
    python scripts/build_all_cache.py --type cir --dataset roxford5k --data-dir data/roxford5k

    # 强制重建
    python scripts/build_all_cache.py --type all --force
"""

import argparse
import sys
import time
from pathlib import Path

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def build_dcmh_cache(datasets: list, force: bool = False, use_mat: bool = True):
    """
    构建 DCMH 缓存。

    包含：
    - 图像哈希码（用于文本→图像检索）
    - 文本哈希码（用于图像→文本检索）
    - 加密数据库

    参数：
        datasets: 数据集列表，如 ['flickr25k', 'nuswide']
        force: 是否强制重建
        use_mat: 是否使用 .mat 文件构建 flickr25k 缓存
    """
    print("\n" + "=" * 60)
    print("构建 DCMH 缓存（完整版）")
    print("=" * 60)

    from backend.app.services.dcmh_service import get_dcmh_service
    from backend.app.services.aspe_service import get_aspe_service
    from backend.app.services.dataset_service import get_dataset_service
    from backend.app.services.hash_cache_service import get_hash_cache_service

    results = {}

    for dataset in datasets:
        print(f"\n--- 处理数据集: {dataset} ---")
        start_time = time.time()

        try:
            # 初始化服务
            dcmh_service = get_dcmh_service(dataset=dataset)
            dataset_service = get_dataset_service(dataset_name=dataset)
            aspe_service = get_aspe_service()
            hash_cache = get_hash_cache_service()

            # 检查模型是否加载
            if not dcmh_service.is_loaded():
                print(f"  警告: {dataset} 的 DCMH 模型未加载，跳过")
                results[dataset] = {"success": False, "error": "模型未加载"}
                continue

            # 构建完整的数据库缓存
            if dataset == 'flickr25k' and use_mat:
                # 使用 .mat 文件构建（确保与训练数据一致）
                print(f"  正在从 .mat 文件构建数据库缓存...")
                image_codes, text_codes, labels = hash_cache.build_from_mat(
                    dcmh_service,
                    dataset=dataset,
                    force_rebuild=force
                )
            else:
                # 使用原始数据构建
                print(f"  正在构建完整数据库缓存...")
                image_codes, text_codes, labels = hash_cache.build_full_database_cache(
                    dcmh_service, dataset_service,
                    batch_size=32, force_rebuild=force, dataset=dataset
                )
            print(f"  图像哈希码: {image_codes.shape}")
            print(f"  文本哈希码: {text_codes.shape}")

            # 构建查询集缓存（仅 flickr25k 需要从原始数据构建）
            if dataset == 'flickr25k':
                print(f"  正在构建查询集缓存...")
                query_codes, query_labels = hash_cache.build_query_cache(
                    dcmh_service, dataset_service,
                    batch_size=32, force_rebuild=force, dataset=dataset
                )
                print(f"  查询集大小: {query_codes.shape[0]} 条记录")

            # 构建加密缓存
            print(f"  正在构建加密缓存...")
            hash_cache.build_encrypted_cache(aspe_service, force_rebuild=force, dataset=dataset)

            elapsed = time.time() - start_time
            results[dataset] = {
                "success": True,
                "database_size": image_codes.shape[0],
                "image_codes_shape": image_codes.shape,
                "text_codes_shape": text_codes.shape,
                "encrypted_cached": True,
                "elapsed_seconds": elapsed
            }
            print(f"  完成! 耗时: {elapsed:.2f} 秒")

        except Exception as e:
            elapsed = time.time() - start_time
            results[dataset] = {"success": False, "error": str(e), "elapsed_seconds": elapsed}
            print(f"  失败: {e}")
            import traceback
            traceback.print_exc()

    # 打印摘要
    print("\n" + "-" * 60)
    print("DCMH 缓存构建摘要:")
    for dataset, result in results.items():
        if result["success"]:
            print(f"  {dataset}: 成功")
            print(f"    - 图像哈希码: {result['image_codes_shape']}")
            print(f"    - 文本哈希码: {result['text_codes_shape']}")
            print(f"    - 加密缓存: 已构建")
            print(f"    - 耗时: {result['elapsed_seconds']:.2f}s")
        else:
            print(f"  {dataset}: 失败 - {result.get('error', '未知错误')}")

    return results


def build_cir_cache(datasets: list, data_dirs: dict = None, force: bool = False):
    """
    构建 CIR 缓存。

    包含：
    - 明文特征
    - 加密特征
    - 持久化密钥

    参数：
        datasets: 数据集列表，如 ['roxford5k', 'rparis6k']
        data_dirs: 数据目录映射，如 {'roxford5k': 'data/roxford5k'}
        force: 是否强制重建
    """
    print("\n" + "=" * 60)
    print("构建 CIR 缓存")
    print("=" * 60)

    from backend.app.services.cir_service import get_cir_service
    from backend.app.services.hash_cache_service import get_hash_cache_service
    from config.dataset_config import get_cir_dataset_config, check_cir_dataset_exists

    results = {}
    data_dirs = data_dirs or {}

    # 使用 dataset_config.py 中的正确路径配置
    default_data_dirs = {}
    for dataset_name in ['roxford5k', 'rparis6k']:
        try:
            config = get_cir_dataset_config(dataset_name)
            # images_dir 指向 jpg 目录，我们需要其父目录（包含 jpg 和 gnd 文件）
            default_data_dirs[dataset_name] = str(config['images_dir'])
        except Exception as e:
            print(f"  警告: 无法获取 {dataset_name} 配置: {e}")

    for dataset in datasets:
        print(f"\n--- 处理数据集: {dataset} ---")
        start_time = time.time()

        try:
            # 获取图像目录（优先使用命令行参数，其次使用配置文件）
            image_dir = data_dirs.get(dataset) or default_data_dirs.get(dataset)
            if image_dir is None:
                print(f"  跳过: 未找到数据集配置")
                results[dataset] = {"success": False, "error": "未找到数据集配置"}
                continue

            image_dir = Path(image_dir)

            # 使用 check_cir_dataset_exists 验证数据集
            exists, path, message = check_cir_dataset_exists(dataset)
            if not exists:
                print(f"  跳过: {message}")
                results[dataset] = {"success": False, "error": message}
                continue

            print(f"  数据集路径: {path}")
            print(f"  图像目录: {image_dir}")

            # 初始化服务
            cir_service = get_cir_service()
            hash_cache = get_hash_cache_service()

            # 检查模型是否加载
            if cir_service.model is None:
                model_path = PROJECT_ROOT / 'data' / 'networks' / 'gl18-tl-resnet101-gem-w-a4d43db.pth'
                if model_path.exists():
                    print(f"  正在加载 CIR 模型...")
                    cir_service.load_model(str(model_path))
                else:
                    print(f"  警告: CIR 模型未找到，跳过")
                    results[dataset] = {"success": False, "error": "模型未加载"}
                    continue

            # 构建特征缓存
            print(f"  正在构建特征缓存...")
            features, image_names = hash_cache.build_cir_cache(
                cir_service=cir_service,
                image_dir=str(image_dir),
                dataset=dataset,
                force_rebuild=force
            )

            # 构建加密缓存
            print(f"  正在构建加密缓存...")
            # 确保特征已加密
            if cir_service.db_encrypted_features is not None:
                import torch
                encrypted = cir_service.db_encrypted_features
                if isinstance(encrypted, torch.Tensor):
                    encrypted = encrypted.numpy()
                hash_cache.save_cir_encrypted(encrypted, dataset)

            elapsed = time.time() - start_time
            results[dataset] = {
                "success": True,
                "database_size": len(image_names),
                "feature_dim": features.shape[1],
                "encrypted_cached": cir_service.db_encrypted_features is not None,
                "elapsed_seconds": elapsed
            }
            print(f"  完成! 数据库大小: {len(image_names)}, 耗时: {elapsed:.2f} 秒")

        except Exception as e:
            elapsed = time.time() - start_time
            results[dataset] = {"success": False, "error": str(e), "elapsed_seconds": elapsed}
            print(f"  失败: {e}")
            import traceback
            traceback.print_exc()

    # 打印摘要
    print("\n" + "-" * 60)
    print("CIR 缓存构建摘要:")
    for dataset, result in results.items():
        if result["success"]:
            print(f"  {dataset}: 成功 - {result['database_size']} 图像 ({result['elapsed_seconds']:.2f}s)")
        else:
            print(f"  {dataset}: 失败 - {result.get('error', '未知错误')}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="统一缓存构建脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 构建所有缓存
  python scripts/build_all_cache.py --type all

  # 仅构建 DCMH 缓存
  python scripts/build_all_cache.py --type dcmh

  # 指定数据集
  python scripts/build_all_cache.py --type dcmh --dataset flickr25k
  python scripts/build_all_cache.py --type cir --dataset roxford5k --data-dir data/roxford5k

  # 强制重建
  python scripts/build_all_cache.py --type all --force
        """
    )

    parser.add_argument(
        "--type",
        choices=["dcmh", "cir", "all"],
        default="all",
        help="缓存类型: dcmh, cir, 或 all (默认: all)"
    )
    parser.add_argument(
        "--dataset",
        help="指定数据集名称 (如 flickr25k, nuswide, roxford5k, rparis6k)"
    )
    parser.add_argument(
        "--data-dir",
        help="CIR 数据集目录 (仅用于 CIR 缓存构建)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="强制重建缓存"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("DeepHash-ASPE 缓存构建工具")
    print("=" * 60)
    print(f"缓存类型: {args.type}")
    print(f"强制重建: {args.force}")
    if args.dataset:
        print(f"指定数据集: {args.dataset}")

    total_start = time.time()
    all_results = {}

    # DCMH 缓存
    if args.type in ["dcmh", "all"]:
        datasets = [args.dataset] if args.dataset and args.dataset in ["flickr25k", "nuswide"] else ["flickr25k", "nuswide"]
        all_results["dcmh"] = build_dcmh_cache(datasets, force=args.force)

    # CIR 缓存
    if args.type in ["cir", "all"]:
        datasets = [args.dataset] if args.dataset and args.dataset in ["roxford5k", "rparis6k"] else ["roxford5k", "rparis6k"]
        data_dirs = {args.dataset: args.data_dir} if args.dataset and args.data_dir else None
        all_results["cir"] = build_cir_cache(datasets, data_dirs=data_dirs, force=args.force)

    total_elapsed = time.time() - total_start

    # 打印总摘要
    print("\n" + "=" * 60)
    print("构建完成!")
    print("=" * 60)
    print(f"总耗时: {total_elapsed:.2f} 秒")

    # 统计成功/失败
    success_count = 0
    fail_count = 0
    for cache_type, results in all_results.items():
        for dataset, result in results.items():
            if result["success"]:
                success_count += 1
            else:
                fail_count += 1

    print(f"成功: {success_count}, 失败: {fail_count}")


if __name__ == "__main__":
    main()