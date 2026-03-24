"""
模型部署脚本：从训练结果目录复制模型到部署目录

使用方法：
    python scripts/deploy_model.py --dataset flickr25k
    python scripts/deploy_model.py --dataset flickr25k --result_dir results/flickr-25k/20260324_090727
"""

import argparse
import shutil
import json
from pathlib import Path
from datetime import datetime

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent

# 数据集映射
DATASET_CONFIG = {
    "flickr25k": {
        "source_pattern": "flickr-25k",
        "target_dir": "data/dcmh/flickr25k",
        "models": ["img_model.pth", "txt_model.pth"],
    }
}


def find_latest_result_dir(dataset: str) -> Path:
    """
    查找指定数据集的最新训练结果目录。

    参数：
        dataset: 数据集名称

    返回：
        最新训练结果目录路径
    """
    config = DATASET_CONFIG.get(dataset)
    if not config:
        raise ValueError(f"不支持的数据集: {dataset}")

    source_pattern = config["source_pattern"]
    results_base = PROJECT_ROOT / "results" / source_pattern

    if not results_base.exists():
        raise FileNotFoundError(f"训练结果目录不存在: {results_base}")

    # 查找所有子目录
    subdirs = [d for d in results_base.iterdir() if d.is_dir()]

    if not subdirs:
        raise FileNotFoundError(f"未找到训练结果: {results_base}")

    # 按修改时间排序，选择最新的
    subdirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    latest = subdirs[0]

    return latest


def get_model_info(model_path: Path) -> dict:
    """
    获取模型文件信息。

    参数：
        model_path: 模型文件路径

    返回：
        包含大小、修改时间等信息的字典
    """
    stat = model_path.stat()
    return {
        "size_mb": round(stat.st_size / (1024 * 1024), 2),
        "modified": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
    }


def deploy_models(
    dataset: str,
    result_dir: Path = None,
    force: bool = False
) -> dict:
    """
    部署模型到 data 目录。

    参数：
        dataset: 数据集名称
        result_dir: 训练结果目录（None 时自动查找最新）
        force: 是否强制覆盖已存在的模型

    返回：
        部署结果信息
    """
    config = DATASET_CONFIG.get(dataset)
    if not config:
        raise ValueError(f"不支持的数据集: {dataset}")

    # 确定源目录
    if result_dir is None:
        result_dir = find_latest_result_dir(dataset)
    else:
        result_dir = Path(result_dir)

    # 目标目录
    target_dir = PROJECT_ROOT / config["target_dir"]
    target_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"模型部署 - {dataset}")
    print(f"{'='*60}")
    print(f"源目录: {result_dir}")
    print(f"目标目录: {target_dir}")
    print()

    deployed = []

    for model_name in config["models"]:
        src_model = result_dir / model_name
        dst_model = target_dir / model_name

        if not src_model.exists():
            print(f"  [跳过] {model_name} - 源文件不存在")
            continue

        if dst_model.exists() and not force:
            print(f"  [跳过] {model_name} - 已存在 (使用 --force 覆盖)")
            continue

        # 复制模型文件
        print(f"  [复制] {model_name}...", end=" ", flush=True)
        info = get_model_info(src_model)
        shutil.copy2(src_model, dst_model)
        print(f"完成 ({info['size_mb']} MB)")

        deployed.append({
            "name": model_name,
            "size_mb": info["size_mb"],
            "source": str(src_model),
            "target": str(dst_model),
        })

    # 复制训练结果信息
    src_result = result_dir / "result.json"
    if src_result.exists():
        dst_result = target_dir / "training_result.json"
        shutil.copy2(src_result, dst_result)
        print(f"  [复制] training_result.json")

    print(f"\n部署完成！共复制 {len(deployed)} 个模型文件。")

    return {
        "dataset": dataset,
        "source_dir": str(result_dir),
        "target_dir": str(target_dir),
        "deployed_models": deployed,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def main():
    parser = argparse.ArgumentParser(description="部署 DCMH 模型到 data 目录")
    parser.add_argument(
        "--dataset",
        type=str,
        default="flickr25k",
        choices=["flickr25k"],
        help="数据集名称 (默认: flickr25k)"
    )
    parser.add_argument(
        "--result_dir",
        type=str,
        default=None,
        help="训练结果目录路径（默认自动查找最新）"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="强制覆盖已存在的模型文件"
    )

    args = parser.parse_args()

    try:
        result = deploy_models(
            dataset=args.dataset,
            result_dir=args.result_dir,
            force=args.force
        )

        # 保存部署记录
        record_file = PROJECT_ROOT / "data" / "dcmh" / args.dataset / "deploy_record.json"
        with open(record_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"部署记录已保存到: {record_file}")

    except Exception as e:
        print(f"\n错误: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())