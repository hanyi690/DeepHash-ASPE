#!/usr/bin/env python
"""
CNN 图像检索预训练模型下载脚本

下载 ResNet101-GeM 等检索模型用于 CIR 服务。
模型来源: https://github.com/filipradenovic/cnnimageretrieval-pytorch
"""

import os
import sys
import argparse
from pathlib import Path
import urllib.request
import hashlib

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 模型下载链接
MODEL_URLS = {
    'resnet101-gem': {
        'url': 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/retrieval-SfM-120k/retrieval-SfM-120k-resnet101-gem-454ad53.pth',
        'desc': 'ResNet101 with GeM pooling, trained on SfM-120k'
    },
    'resnet50-gem': {
        'url': 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/retrieval-SfM-120k/retrieval-SfM-120k-resnet50-gem-f15da7b.pth',
        'desc': 'ResNet50 with GeM pooling, trained on SfM-120k'
    },
    'vgg16-gem': {
        'url': 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/retrieval-SfM-120k/retrieval-SfM-120k-vgg16-gem-eaa6695.pth',
        'desc': 'VGG16 with GeM pooling, trained on SfM-120k'
    },
}

# 白化模型下载链接
WHITENING_URLS = {
    'resnet101-gem': {
        'url': 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/whiten/retrieval-SfM-120k/retrieval-SfM-120k-resnet101-gem-whiten-22ab0c1.pth',
        'desc': 'ResNet101-GeM whitening weights'
    },
    'resnet50-gem': {
        'url': 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/whiten/retrieval-SfM-120k/retrieval-SfM-120k-resnet50-gem-whiten-f15da7b.pth',
        'desc': 'ResNet50-GeM whitening weights'
    },
}


def download_file(url: str, save_path: Path, desc: str = None) -> bool:
    """
    下载文件并显示进度。

    参数:
        url: 下载链接
        save_path: 保存路径
        desc: 文件描述

    返回:
        是否成功下载
    """
    if save_path.exists():
        print(f"[跳过] 文件已存在: {save_path}")
        return True

    save_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[下载] {desc or save_path.name}")
    print(f"  URL: {url}")
    print(f"  目标: {save_path}")

    try:
        # 下载文件
        def report_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                percent = min(100, downloaded * 100 / total_size)
                mb_downloaded = downloaded / (1024 * 1024)
                mb_total = total_size / (1024 * 1024)
                print(f"\r  进度: {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)", end='', flush=True)

        urllib.request.urlretrieve(url, save_path, reporthook=report_progress)
        print("\n[完成] 下载成功")
        return True

    except Exception as e:
        print(f"\n[错误] 下载失败: {e}")
        if save_path.exists():
            save_path.unlink()
        return False


def download_model(
    model_name: str = 'resnet101-gem',
    save_dir: str = None,
    include_whitening: bool = True
) -> Path:
    """
    下载指定的预训练模型。

    参数:
        model_name: 模型名称
        save_dir: 保存目录
        include_whitening: 是否下载白化权重

    返回:
        模型保存路径
    """
    if model_name not in MODEL_URLS:
        print(f"[错误] 未知模型: {model_name}")
        print(f"可用模型: {list(MODEL_URLS.keys())}")
        return None

    if save_dir is None:
        save_dir = PROJECT_ROOT / 'data' / 'models'
    else:
        save_dir = Path(save_dir)

    save_dir.mkdir(parents=True, exist_ok=True)

    model_info = MODEL_URLS[model_name]
    model_path = save_dir / f"{model_name}.pth"

    print(f"\n{'='*60}")
    print(f"下载模型: {model_name}")
    print(f"描述: {model_info['desc']}")
    print(f"{'='*60}\n")

    # 下载主模型
    success = download_file(model_info['url'], model_path, model_info['desc'])

    if not success:
        return None

    # 下载白化权重
    if include_whitening and model_name in WHITENING_URLS:
        whiten_info = WHITENING_URLS[model_name]
        whiten_path = save_dir / f"{model_name}-whiten.pth"

        print(f"\n{'='*60}")
        print("下载白化权重")
        print(f"{'='*60}\n")

        download_file(whiten_info['url'], whiten_path, whiten_info['desc'])

    print(f"\n{'='*60}")
    print("下载完成！")
    print(f"模型路径: {model_path}")
    print(f"{'='*60}\n")

    return model_path


def list_models():
    """列出所有可用的模型。"""
    print("\n可用的预训练模型:\n")
    for name, info in MODEL_URLS.items():
        print(f"  {name}:")
        print(f"    描述: {info['desc']}")
        print()
    print("使用方法:")
    print(f"  python {Path(__file__).name} --model resnet101-gem")
    print(f"  python {Path(__file__).name} --model resnet101-gem --save-dir ./models")


def main():
    parser = argparse.ArgumentParser(
        description='CNN 图像检索预训练模型下载工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python download_cir_model.py                    # 下载默认模型 (resnet101-gem)
  python download_cir_model.py --model resnet50-gem
  python download_cir_model.py --list             # 列出所有可用模型
  python download_cir_model.py --save-dir ./data/models
        """
    )
    parser.add_argument(
        '--model', '-m',
        type=str,
        default='resnet101-gem',
        help='模型名称 (默认: resnet101-gem)'
    )
    parser.add_argument(
        '--save-dir', '-s',
        type=str,
        default=None,
        help='模型保存目录 (默认: data/models)'
    )
    parser.add_argument(
        '--no-whitening',
        action='store_true',
        help='不下载白化权重'
    )
    parser.add_argument(
        '--list', '-l',
        action='store_true',
        help='列出所有可用模型'
    )

    args = parser.parse_args()

    if args.list:
        list_models()
        return

    download_model(
        model_name=args.model,
        save_dir=args.save_dir,
        include_whitening=not args.no_whitening
    )


if __name__ == '__main__':
    main()