#!/usr/bin/env python
"""
CNN 图像检索预训练模型下载脚本

下载 ResNet101-GeM 等检索模型用于 CIR 服务。
模型来源: https://github.com/filipradenovic/cnnimageretrieval-pytorch

模型类型:
- FEATURES: ImageNet 预训练卷积特征（off-the-shelf 模式）
- PRETRAINED: 检索微调完整模型（推荐）
- WHITENING: 后处理白化权重

国内用户推荐使用 Hugging Face 镜像：
- 设置环境变量 HF_ENDPOINT=https://hf-mirror.com
- 或使用 --mirror hf 参数
"""

import os
import sys
import argparse
from pathlib import Path
import urllib.request
import hashlib
import ssl
import socket
import requests
from tqdm import tqdm

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 设置超时
socket.setdefaulttimeout(300)  # 5 分钟超时

# =============================================================================
# 镜像源配置
# =============================================================================

# Hugging Face 镜像（国内推荐）
HF_MIRROR = os.environ.get('HF_ENDPOINT', 'https://hf-mirror.com')

# Hugging Face 模型仓库
HF_REPO = "radenovic/cnnimageretrieval-pytorch"

# =============================================================================
# 模型下载链接
# =============================================================================

# 原始 URL（官方源，国外快）
OFFICIAL_URLS = {
    'gl18-resnet101-gem-w': 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/gl18/gl18-tl-resnet101-gem-w-a4d43db.pth',
    'gl18-resnet50-gem-w': 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/gl18/gl18-tl-resnet50-gem-w-4ec15b1.pth',
    'sfm120k-resnet101-gem': 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/retrieval-SfM-120k/retrievalSfM120k-resnet101-gem-b80fb85.pth',
    'sfm120k-resnet50-gem': 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/retrieval-SfM-120k/retrieval-SfM-120k-resnet50-gem-f15da7b.pth',
    'sfm120k-vgg16-gem': 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/retrieval-SfM-120k/retrieval-SfM-120k-vgg16-gem-eaa6695.pth',
    'resnet50-caffe': 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/imagenet/imagenet-caffe-resnet50-features-ac468af.pth',
    'resnet101-caffe': 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/imagenet/imagenet-caffe-resnet101-features-10a101d.pth',
    'resnet101-gem-whiten': 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/whiten/retrieval-SfM-120k/retrieval-SfM-120k-resnet101-gem-whiten-22ab0c1.pth',
    'resnet50-gem-whiten': 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/whiten/retrieval-SfM-120k/retrieval-SfM-120k-resnet50-gem-whiten-f15da7b.pth',
}

# Hugging Face URL 模板（国内镜像快）
HF_URL_TEMPLATE = "{mirror}/{repo}/resolve/main/{filename}"

# ImageNet 预训练特征模型（FEATURES）- 用于 off-the-shelf 模式
FEATURES_URLS = {
    'resnet50-caffe': {
        'filename': 'imagenet-caffe-resnet50-features-ac468af.pth',
        'desc': 'ResNet50 ImageNet Caffe 特征',
        'size': '~90MB'
    },
    'resnet101-caffe': {
        'filename': 'imagenet-caffe-resnet101-features-10a101d.pth',
        'desc': 'ResNet101 ImageNet Caffe 特征',
        'size': '~170MB'
    },
}

# 检索微调模型（PRETRAINED）- 推荐使用
PRETRAINED_URLS = {
    # GL18 系列（推荐，性能最高）
    'gl18-resnet101-gem-w': {
        'filename': 'gl18-tl-resnet101-gem-w-a4d43db.pth',
        'desc': 'GL18 ResNet101-GeM-W（推荐，最高性能）',
        'size': '~340MB',
        'performance': 'ROxf(M): 67.3, RPar(M): 80.6'
    },
    'gl18-resnet50-gem-w': {
        'filename': 'gl18-tl-resnet50-gem-w-4ec15b1.pth',
        'desc': 'GL18 ResNet50-GeM-W',
        'size': '~100MB',
        'performance': 'ROxf(M): 62.0, RPar(M): 76.0'
    },
    # SfM-120k 系列
    'sfm120k-resnet101-gem': {
        'filename': 'retrievalSfM120k-resnet101-gem-b80fb85.pth',
        'desc': 'SfM-120k ResNet101-GeM',
        'size': '~170MB',
        'performance': 'ROxf(M): 65.4, RPar(M): 76.7'
    },
    'sfm120k-resnet50-gem': {
        'filename': 'retrieval-SfM-120k-resnet50-gem-f15da7b.pth',
        'desc': 'SfM-120k ResNet50-GeM',
        'size': '~100MB',
        'performance': 'ROxf(M): 60.0, RPar(M): 72.0'
    },
    'sfm120k-vgg16-gem': {
        'filename': 'retrieval-SfM-120k-vgg16-gem-eaa6695.pth',
        'desc': 'SfM-120k VGG16-GeM',
        'size': '~550MB',
        'performance': 'ROxf(M): 57.0, RPar(M): 70.0'
    },
}

# 白化权重（WHITENING）- 用于 off-the-shelf 模式
WHITENING_URLS = {
    'resnet101-gem': {
        'filename': 'retrieval-SfM-120k-resnet101-gem-whiten-22ab0c1.pth',
        'desc': 'ResNet101-GeM 白化权重'
    },
    'resnet50-gem': {
        'filename': 'retrieval-SfM-120k-resnet50-gem-whiten-f15da7b.pth',
        'desc': 'ResNet50-GeM 白化权重'
    },
}

# 兼容旧版模型名称映射
LEGACY_MODEL_MAP = {
    'resnet101-gem': 'sfm120k-resnet101-gem',
    'resnet50-gem': 'sfm120k-resnet50-gem',
    'vgg16-gem': 'sfm120k-vgg16-gem',
}


def download_file(url: str, save_path: Path, desc: str = None) -> bool:
    """
    下载文件并显示进度（使用 requests 库，支持断点续传）。

    参数:
        url: 下载链接
        save_path: 保存路径
        desc: 文件描述

    返回:
        是否成功下载
    """
    if save_path.exists():
        existing_size = save_path.stat().st_size
        # 检查文件是否完整（尝试获取远程文件大小）
        try:
            head_resp = requests.head(url, timeout=30, verify=False, allow_redirects=True)
            total_size = int(head_resp.headers.get('content-length', 0))
            if total_size > 0 and existing_size >= total_size * 0.99:
                print(f"[跳过] 文件已存在: {save_path}")
                return True
        except:
            pass  # 无法获取远程大小，继续下载

    save_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[下载] {desc or save_path.name}")
    print(f"  URL: {url}")
    print(f"  目标: {save_path}")

    try:
        # 断点续传
        existing_size = save_path.stat().st_size if save_path.exists() else 0
        headers = {}
        if existing_size > 0:
            headers['Range'] = f'bytes={existing_size}-'
            print(f"  断点续传: 从 {existing_size / (1024*1024):.1f} MB 继续")

        # 禁用 SSL 验证
        requests.packages.urllib3.disable_warnings()

        response = requests.get(
            url,
            stream=True,
            timeout=300,
            verify=False,
            headers=headers
        )
        response.raise_for_status()

        # 获取总大小
        total_size = int(response.headers.get('content-length', 0))
        if existing_size > 0 and 'content-range' in response.headers:
            # 断点续传时，总大小从 content-range 解析
            content_range = response.headers['content-range']
            total_size = int(content_range.split('/')[-1])
        total_size += existing_size

        # 进度条
        mode = 'ab' if existing_size > 0 else 'wb'
        with open(save_path, mode) as f:
            with tqdm(
                total=total_size,
                initial=existing_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
                desc="  进度"
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

        print("\n[完成] 下载成功")
        return True

    except Exception as e:
        print(f"\n[错误] 下载失败: {e}")
        print(f"  提示: 可以稍后重试，脚本支持断点续传")
        return False


def get_model_info(model_name: str, mirror: str = None) -> dict:
    """
    获取模型信息。

    参数:
        model_name: 模型名称
        mirror: 镜像源 ('hf' 使用 Hugging Face 镜像)

    返回:
        模型信息字典，包含 'type', 'filename', 'url', 'desc' 等
    """
    # 兼容旧版名称
    if model_name in LEGACY_MODEL_MAP:
        model_name = LEGACY_MODEL_MAP[model_name]

    info = None
    if model_name in PRETRAINED_URLS:
        info = PRETRAINED_URLS[model_name].copy()
        info['type'] = 'pretrained'
    elif model_name in FEATURES_URLS:
        info = FEATURES_URLS[model_name].copy()
        info['type'] = 'features'
    else:
        return None

    # 生成 URL
    filename = info['filename']

    if mirror == 'hf':
        # 使用 Hugging Face 镜像
        info['url'] = HF_URL_TEMPLATE.format(
            mirror=HF_MIRROR,
            repo=HF_REPO,
            filename=filename
        )
        info['mirror'] = 'Hugging Face 镜像'
    else:
        # 使用官方源
        info['url'] = OFFICIAL_URLS.get(model_name, '')
        info['mirror'] = '官方源'

    return info


def download_model(
    model_name: str = 'gl18-resnet101-gem-w',
    save_dir: str = None,
    include_whitening: bool = True,
    mirror: str = None
) -> Path:
    """
    下载指定的预训练模型。

    参数:
        model_name: 模型名称
        save_dir: 保存目录
        include_whitening: 是否下载白化权重
        mirror: 镜像源 ('hf' 使用 Hugging Face 镜像，国内推荐)

    返回:
        模型保存路径
    """
    model_info = get_model_info(model_name, mirror)

    if model_info is None:
        print(f"[错误] 未知模型: {model_name}")
        print(f"\n可用模型:")
        print(f"  预训练模型（推荐）: {list(PRETRAINED_URLS.keys())}")
        print(f"  特征模型: {list(FEATURES_URLS.keys())}")
        return None

    if save_dir is None:
        save_dir = PROJECT_ROOT / 'data' / 'networks'
    else:
        save_dir = Path(save_dir)

    save_dir.mkdir(parents=True, exist_ok=True)

    # 确定文件名
    filename = model_info['filename']
    model_path = save_dir / filename

    print(f"\n{'='*60}")
    print(f"下载模型: {model_name}")
    print(f"镜像: {model_info['mirror']}")
    print(f"类型: {model_info['type']}")
    print(f"描述: {model_info['desc']}")
    if 'size' in model_info:
        print(f"大小: {model_info['size']}")
    if 'performance' in model_info:
        print(f"性能: {model_info['performance']}")
    print(f"{'='*60}\n")

    # 下载主模型
    success = download_file(model_info['url'], model_path, model_info['desc'])

    if not success:
        return None

    # 下载白化权重（仅对 features 模型有意义）
    if include_whitening and model_info['type'] == 'features':
        # 从模型名称推断白化权重
        if 'resnet101' in model_name:
            whiten_key = 'resnet101-gem'
        elif 'resnet50' in model_name:
            whiten_key = 'resnet50-gem'
        else:
            whiten_key = None

        if whiten_key and whiten_key in WHITENING_URLS:
            whiten_info = WHITENING_URLS[whiten_key].copy()
            whiten_dir = PROJECT_ROOT / 'data' / 'whiten'
            whiten_dir.mkdir(parents=True, exist_ok=True)

            # 生成白化权重 URL
            if mirror == 'hf':
                whiten_url = HF_URL_TEMPLATE.format(
                    mirror=HF_MIRROR,
                    repo=HF_REPO,
                    filename=whiten_info['filename']
                )
            else:
                whiten_url = OFFICIAL_URLS.get(f'{whiten_key}-whiten', '')

            whiten_path = whiten_dir / whiten_info['filename']

            print(f"\n{'='*60}")
            print("下载白化权重")
            print(f"{'='*60}\n")

            download_file(whiten_url, whiten_path, whiten_info['desc'])

    print(f"\n{'='*60}")
    print("下载完成！")
    print(f"模型路径: {model_path}")
    print(f"{'='*60}\n")

    return model_path


def list_models():
    """列出所有可用的模型。"""
    print("\n" + "="*60)
    print("可用的预训练模型（PRETRAINED，推荐）")
    print("="*60)
    for name, info in PRETRAINED_URLS.items():
        print(f"\n  {name}:")
        print(f"    描述: {info['desc']}")
        if 'size' in info:
            print(f"    大小: {info['size']}")
        if 'performance' in info:
            print(f"    性能: {info['performance']}")

    print("\n" + "="*60)
    print("ImageNet 特征模型（FEATURES，用于 off-the-shelf 模式）")
    print("="*60)
    for name, info in FEATURES_URLS.items():
        print(f"\n  {name}:")
        print(f"    描述: {info['desc']}")
        if 'size' in info:
            print(f"    大小: {info['size']}")

    print("\n" + "="*60)
    print("白化权重（WHITENING）")
    print("="*60)
    for name, info in WHITENING_URLS.items():
        print(f"\n  {name}:")
        print(f"    描述: {info['desc']}")

    print("\n使用方法:")
    print(f"  python {Path(__file__).name} --model gl18-resnet101-gem-w  # 推荐")
    print(f"  python {Path(__file__).name} --model resnet101-caffe --type features")
    print(f"  python {Path(__file__).name} --list")


def main():
    parser = argparse.ArgumentParser(
        description='CNN 图像检索预训练模型下载工具（国内用户推荐使用 --mirror hf）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python download_cir_model.py --mirror hf                   # 国内推荐：使用 HF 镜像下载
  python download_cir_model.py --model gl18-resnet101-gem-w --mirror hf
  python download_cir_model.py --model sfm120k-resnet101-gem  # SfM-120k 模型
  python download_cir_model.py --model resnet101-caffe --mirror hf
  python download_cir_model.py --list                         # 列出所有模型

国内镜像:
  设置环境变量可切换镜像: set HF_ENDPOINT=https://hf-mirror.com
  或使用参数: --mirror hf
        """
    )
    parser.add_argument(
        '--model', '-m',
        type=str,
        default='gl18-resnet101-gem-w',
        help='模型名称 (默认: gl18-resnet101-gem-w)'
    )
    parser.add_argument(
        '--save-dir', '-s',
        type=str,
        default=None,
        help='模型保存目录 (默认: data/networks)'
    )
    parser.add_argument(
        '--no-whitening',
        action='store_true',
        help='不下载白化权重'
    )
    parser.add_argument(
        '--mirror',
        type=str,
        choices=['hf'],
        default=None,
        help='镜像源: hf=Hugging Face 镜像（国内推荐）'
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
        include_whitening=not args.no_whitening,
        mirror=args.mirror
    )


if __name__ == '__main__':
    main()