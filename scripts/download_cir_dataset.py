#!/usr/bin/env python
"""
CNN 图像检索数据集下载脚本

下载 ROxford5k 和 RParis6k 数据集用于 CIR 评估。
数据集来源: https://github.com/filipradenovic/cnnimageretrieval-pytorch

支持断点续传，下载中断后重新运行即可继续。
支持 HuggingFace 镜像下载，解决国内网络问题。
"""

import os
import sys
import argparse
import tarfile
import zipfile
from pathlib import Path
import pickle
import shutil
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
# HuggingFace 镜像配置
# =============================================================================

# HuggingFace 数据集配置
HF_DATASETS = {
    'roxford5k': {
        'repo': 'randall-lab/revisitop',
        'files': {
            'images': 'roxford5k/jpg',
            'gnd': 'roxford5k/gnd_roxford5k.pkl'
        }
    },
    'rparis6k': {
        'repo': 'randall-lab/revisitop',
        'files': {
            'images': 'rparis6k/jpg',
            'gnd': 'rparis6k/gnd_rparis6k.pkl'
        }
    }
}

# =============================================================================
# 数据集下载链接
# =============================================================================

# 国内镜像（如果可用）
# 注意：这些数据集目前没有官方国内镜像，建议使用代理或手动下载
MIRROR_INFO = """
数据集下载加速建议：
1. 使用 HuggingFace 镜像: --mirror hf
2. 使用代理：设置 HTTP_PROXY 和 HTTPS_PROXY 环境变量
3. 手动下载：使用浏览器/下载工具下载后放到对应目录
   - Oxford5k: https://www.robots.ox.ac.uk/~vgg/data/oxbuildings/oxbuild_images-v1.tgz
   - Paris6k: https://www.robots.ox.ac.uk/~vgg/data/parisbuildings/
   - Ground Truth: http://cmp.felk.cvut.cz/cnnimageretrieval/data/test/
"""

DATASET_URLS = {
    # Oxford5k 图像（ROxford5k 使用相同图像）
    'oxford5k': {
        'images': 'https://www.robots.ox.ac.uk/~vgg/data/oxbuildings/oxbuild_images-v1.tgz',
        'size': '~1.8GB',
        'desc': 'Oxford Buildings 5k 数据集'
    },
    # Paris6k 图像（RParis6k 使用相同图像）
    'paris6k': {
        'images': [
            'https://www.robots.ox.ac.uk/~vgg/data/parisbuildings/paris_1-v1.tgz',
            'https://www.robots.ox.ac.uk/~vgg/data/parisbuildings/paris_2-v1.tgz'
        ],
        'size': '~2.2GB',
        'desc': 'Paris Buildings 6k 数据集'
    },
    # Ground Truth 文件
    'gnd_roxford5k': {
        'url': 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/test/roxford5k/gnd_roxford5k.pkl',
        'size': '~500KB',
        'desc': 'ROxford5k Ground Truth'
    },
    'gnd_rparis6k': {
        'url': 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/test/rparis6k/gnd_rparis6k.pkl',
        'size': '~600KB',
        'desc': 'RParis6k Ground Truth'
    },
}

# 数据集信息
DATASET_INFO = {
    'roxford5k': {
        'images': 5062,
        'queries': 70,
        'source': 'Oxford Buildings',
        'note': 'Revisited 版本，包含更准确的标注'
    },
    'rparis6k': {
        'images': 6392,
        'queries': 70,
        'source': 'Paris Buildings',
        'note': 'Revisited 版本，包含更准确的标注'
    }
}


def download_file(url: str, save_path: Path, desc: str = None) -> bool:
    """
    下载文件并显示进度（支持断点续传）。

    参数:
        url: 下载链接
        save_path: 保存路径
        desc: 文件描述

    返回:
        是否成功下载
    """
    # 检查文件是否已完整
    if save_path.exists():
        existing_size = save_path.stat().st_size
        try:
            requests.packages.urllib3.disable_warnings()
            head_resp = requests.head(url, timeout=30, verify=False, allow_redirects=True)
            total_size = int(head_resp.headers.get('content-length', 0))
            if total_size > 0 and existing_size >= total_size * 0.99:
                print(f"[跳过] 文件已存在: {save_path}")
                return True
        except:
            pass

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
            content_range = response.headers['content-range']
            total_size = int(content_range.split('/')[-1])
        total_size += existing_size

        # 进度条下载
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
        print(f"  提示: 重新运行可断点续传")
        return False


def download_from_huggingface(dataset_name: str, data_dir: Path) -> bool:
    """
    从 HuggingFace 镜像下载数据集。

    参数:
        dataset_name: 数据集名称 ('roxford5k' 或 'rparis6k')
        data_dir: 数据保存目录

    返回:
        是否成功下载
    """
    try:
        from huggingface_hub import snapshot_download, list_repo_files, hf_hub_download
        from huggingface_hub.utils import RepositoryNotFoundError
    except ImportError:
        print("[错误] 需要安装 huggingface_hub 库")
        print("  请运行: pip install huggingface_hub")
        return False

    hf_config = HF_DATASETS.get(dataset_name)
    if not hf_config:
        print(f"[错误] 不支持的数据集: {dataset_name}")
        return False

    print(f"\n[下载] 从 HuggingFace 镜像下载 {dataset_name}")
    print(f"  仓库: {hf_config['repo']}")

    try:
        # 设置 HuggingFace 镜像（国内加速）
        hf_endpoint = os.environ.get('HF_ENDPOINT', 'https://huggingface.co')

        print(f"  正在检查数据集内容... (使用镜像: {hf_endpoint})")

        # 首先检查数据集是否包含图像文件
        try:
            files = list(list_repo_files(hf_config['repo'], repo_type='dataset'))
            images_remote = hf_config['files']['images']
            image_files = [f for f in files if f.startswith(images_remote) and f.endswith('.jpg')]

            if len(image_files) == 0:
                print(f"[警告] HuggingFace 数据集不包含图像文件")
                print(f"  提示: 请使用 --mirror official 或设置代理后重试")
                return False

            print(f"  找到 {len(image_files)} 张图像")

        except Exception as e:
            print(f"[错误] 无法获取数据集文件列表: {e}")
            return False

        # 创建临时目录
        local_dir = data_dir / '.hf_temp'
        local_dir.mkdir(parents=True, exist_ok=True)

        images_local = data_dir / 'jpg'
        images_local.mkdir(parents=True, exist_ok=True)

        gnd_remote = hf_config['files']['gnd']
        gnd_local = data_dir / f'gnd_{dataset_name}.pkl'

        success = True

        # 下载 Ground Truth
        print(f"\n[下载] Ground Truth 文件...")
        try:
            hf_hub_download(
                repo_id=hf_config['repo'],
                filename=gnd_remote,
                local_dir=str(local_dir),
                repo_type='dataset'
            )
            # 移动到目标位置
            gnd_temp = local_dir / gnd_remote
            if gnd_temp.exists():
                shutil.move(str(gnd_temp), str(gnd_local))
                print(f"  已保存到: {gnd_local}")
        except Exception as e:
            print(f"[错误] Ground Truth 下载失败: {e}")
            success = False

        # 下载图像（需要逐个下载）
        print(f"\n[下载] 图像文件...")
        downloaded = 0
        for img_file in tqdm(image_files, desc="  下载图像"):
            try:
                local_path = local_dir / img_file
                if not local_path.exists():
                    hf_hub_download(
                        repo_id=hf_config['repo'],
                        filename=img_file,
                        local_dir=str(local_dir),
                        repo_type='dataset'
                    )
                downloaded += 1
            except Exception:
                pass  # 忽略单个文件下载失败

        # 移动图像到目标目录
        temp_images_dir = local_dir / images_remote
        if temp_images_dir.exists():
            for img in temp_images_dir.glob('*.jpg'):
                shutil.move(str(img), str(images_local / img.name))

        # 递归查找所有 jpg 文件
        for img in local_dir.rglob('*.jpg'):
            if img.parent != images_local:
                shutil.move(str(img), str(images_local / img.name))

        image_count = len(list(images_local.glob('*.jpg')))
        print(f"  已下载 {image_count} 张图像")

        # 清理临时目录
        shutil.rmtree(local_dir, ignore_errors=True)

        if image_count < 100:  # 图像太少，认为下载失败
            print(f"[警告] 下载的图像数量不足 ({image_count})")
            return False

        print("\n[完成] HuggingFace 下载成功")
        return True

    except RepositoryNotFoundError:
        print(f"[错误] HuggingFace 仓库不存在: {hf_config['repo']}")
        return False
    except Exception as e:
        print(f"[错误] HuggingFace 下载失败: {e}")
        return False


def extract_tgz(tgz_path: Path, target_dir: Path) -> bool:
    """
    解压 tar.gz 文件。

    参数:
        tgz_path: 压缩文件路径
        target_dir: 目标目录

    返回:
        是否成功解压
    """
    print(f"[解压] {tgz_path.name} -> {target_dir}")

    try:
        target_dir.mkdir(parents=True, exist_ok=True)

        with tarfile.open(tgz_path, 'r:gz') as tar:
            tar.extractall(path=target_dir)

        print("[完成] 解压成功")
        return True

    except Exception as e:
        print(f"[错误] 解压失败: {e}")
        return False


def download_roxford5k(data_dir: Path = None, keep_archive: bool = False, mirror: str = 'auto') -> bool:
    """
    下载 ROxford5k 数据集。

    参数:
        data_dir: 数据目录
        keep_archive: 是否保留压缩包
        mirror: 下载源 ('auto', 'hf', 'official')

    返回:
        是否成功下载
    """
    if data_dir is None:
        data_dir = PROJECT_ROOT / 'data' / 'test' / 'roxford5k'
    else:
        data_dir = Path(data_dir)

    jpg_dir = data_dir / 'jpg'
    gnd_path = data_dir / 'gnd_roxford5k.pkl'

    # 检查是否已存在
    if jpg_dir.exists() and gnd_path.exists():
        image_count = len(list(jpg_dir.glob('*.jpg')))
        if image_count > 5000:  # Oxford5k 有 5062 张图
            print(f"[跳过] ROxford5k 已存在: {data_dir}")
            return True

    print(f"\n{'='*60}")
    print("下载 ROxford5k 数据集")
    print(f"{'='*60}\n")

    # 根据 mirror 参数选择下载源
    if mirror == 'hf':
        # 使用 HuggingFace 镜像
        print("[信息] 使用 HuggingFace 镜像下载")
        return download_from_huggingface('roxford5k', data_dir)

    elif mirror == 'official':
        # 使用官方源
        print("[信息] 使用官方源下载")
        return _download_roxford5k_official(data_dir, keep_archive)

    else:  # auto
        # 自动选择：优先尝试 HuggingFace，失败后尝试官方源
        print("[信息] 自动选择下载源")

        # 先尝试 HuggingFace
        print("\n[尝试] HuggingFace 镜像...")
        if download_from_huggingface('roxford5k', data_dir):
            return True

        # HuggingFace 失败，尝试官方源
        print("\n[回退] HuggingFace 失败，尝试官方源...")
        return _download_roxford5k_official(data_dir, keep_archive)


def _download_roxford5k_official(data_dir: Path, keep_archive: bool = False) -> bool:
    """
    从官方源下载 ROxford5k 数据集。

    参数:
        data_dir: 数据目录
        keep_archive: 是否保留压缩包

    返回:
        是否成功下载
    """
    jpg_dir = data_dir / 'jpg'
    gnd_path = data_dir / 'gnd_roxford5k.pkl'

    # 创建临时目录
    temp_dir = data_dir / '.temp'
    temp_dir.mkdir(parents=True, exist_ok=True)

    success = True

    # 下载 Oxford5k 图像
    archive_path = temp_dir / 'oxbuild_images-v1.tgz'
    if not download_file(DATASET_URLS['oxford5k']['images'], archive_path, 'Oxford5k 图像'):
        success = False
    else:
        # 解压图像
        if not extract_tgz(archive_path, temp_dir):
            success = False
        else:
            # 移动图像到正确位置
            # Oxford5k 解压后的目录结构可能是 oxbuild_images/ 或直接是图像文件
            extracted_dirs = list(temp_dir.glob('oxbuild_images'))
            if extracted_dirs:
                source_dir = extracted_dirs[0]
            else:
                # 查找包含 jpg 的目录
                for d in temp_dir.iterdir():
                    if d.is_dir() and list(d.glob('*.jpg')):
                        source_dir = d
                        break
                else:
                    source_dir = temp_dir

            # 确保目标目录存在
            jpg_dir.mkdir(parents=True, exist_ok=True)

            # 移动图像
            image_files = list(source_dir.glob('*.jpg'))
            for img in image_files:
                shutil.move(str(img), str(jpg_dir / img.name))

            print(f"[信息] 已移动 {len(image_files)} 张图像到 {jpg_dir}")

    # 下载 Ground Truth
    if not download_file(DATASET_URLS['gnd_roxford5k']['url'], gnd_path, 'ROxford5k Ground Truth'):
        success = False

    # 清理临时文件
    if not keep_archive:
        shutil.rmtree(temp_dir, ignore_errors=True)

    if success:
        print(f"\n{'='*60}")
        print("ROxford5k 下载完成！")
        print(f"图像目录: {jpg_dir}")
        print(f"Ground Truth: {gnd_path}")
        print(f"{'='*60}\n")

    return success


def download_rparis6k(data_dir: Path = None, keep_archive: bool = False, mirror: str = 'auto') -> bool:
    """
    下载 RParis6k 数据集。

    参数:
        data_dir: 数据目录
        keep_archive: 是否保留压缩包
        mirror: 下载源 ('auto', 'hf', 'official')

    返回:
        是否成功下载
    """
    if data_dir is None:
        data_dir = PROJECT_ROOT / 'data' / 'test' / 'rparis6k'
    else:
        data_dir = Path(data_dir)

    jpg_dir = data_dir / 'jpg'
    gnd_path = data_dir / 'gnd_rparis6k.pkl'

    # 检查是否已存在
    if jpg_dir.exists() and gnd_path.exists():
        image_count = len(list(jpg_dir.glob('*.jpg')))
        if image_count > 6000:  # Paris6k 有 6392 张图
            print(f"[跳过] RParis6k 已存在: {data_dir}")
            return True

    print(f"\n{'='*60}")
    print("下载 RParis6k 数据集")
    print(f"{'='*60}\n")

    # 根据 mirror 参数选择下载源
    if mirror == 'hf':
        # 使用 HuggingFace 镜像
        print("[信息] 使用 HuggingFace 镜像下载")
        return download_from_huggingface('rparis6k', data_dir)

    elif mirror == 'official':
        # 使用官方源
        print("[信息] 使用官方源下载")
        return _download_rparis6k_official(data_dir, keep_archive)

    else:  # auto
        # 自动选择：优先尝试 HuggingFace，失败后尝试官方源
        print("[信息] 自动选择下载源")

        # 先尝试 HuggingFace
        print("\n[尝试] HuggingFace 镜像...")
        if download_from_huggingface('rparis6k', data_dir):
            return True

        # HuggingFace 失败，尝试官方源
        print("\n[回退] HuggingFace 失败，尝试官方源...")
        return _download_rparis6k_official(data_dir, keep_archive)


def _download_rparis6k_official(data_dir: Path, keep_archive: bool = False) -> bool:
    """
    从官方源下载 RParis6k 数据集。

    参数:
        data_dir: 数据目录
        keep_archive: 是否保留压缩包

    返回:
        是否成功下载
    """
    jpg_dir = data_dir / 'jpg'
    gnd_path = data_dir / 'gnd_rparis6k.pkl'

    # 创建临时目录
    temp_dir = data_dir / '.temp'
    temp_dir.mkdir(parents=True, exist_ok=True)

    success = True

    # 下载 Paris6k 图像（分两个压缩包）
    for i, url in enumerate(DATASET_URLS['paris6k']['images'], 1):
        archive_path = temp_dir / f'paris_{i}-v1.tgz'
        if not download_file(url, archive_path, f'Paris6k 图像 ({i}/2)'):
            success = False
            continue

        # 解压图像
        if not extract_tgz(archive_path, temp_dir):
            success = False
            continue

    if success:
        # 移动图像到正确位置
        jpg_dir.mkdir(parents=True, exist_ok=True)

        # Paris6k 解压后可能有 parismap_jpg 目录
        all_jpgs = list(temp_dir.rglob('*.jpg'))
        for img in all_jpgs:
            if img.parent != jpg_dir:
                shutil.move(str(img), str(jpg_dir / img.name))

        print(f"[信息] 已移动 {len(all_jpgs)} 张图像到 {jpg_dir}")

    # 下载 Ground Truth
    if not download_file(DATASET_URLS['gnd_rparis6k']['url'], gnd_path, 'RParis6k Ground Truth'):
        success = False

    # 清理临时文件
    if not keep_archive:
        shutil.rmtree(temp_dir, ignore_errors=True)

    if success:
        print(f"\n{'='*60}")
        print("RParis6k 下载完成！")
        print(f"图像目录: {jpg_dir}")
        print(f"Ground Truth: {gnd_path}")
        print(f"{'='*60}\n")

    return success


def verify_dataset(dataset_name: str, data_dir: Path = None) -> dict:
    """
    验证数据集完整性。

    参数:
        dataset_name: 数据集名称
        data_dir: 数据目录

    返回:
        验证结果字典
    """
    if data_dir is None:
        data_dir = PROJECT_ROOT / 'data' / 'test' / dataset_name
    else:
        data_dir = Path(data_dir)

    jpg_dir = data_dir / 'jpg'
    gnd_file = data_dir / f'gnd_{dataset_name}.pkl'

    result = {
        'name': dataset_name,
        'path': str(data_dir),
        'images_exist': False,
        'gnd_exist': False,
        'image_count': 0,
        'valid': False
    }

    # 检查图像
    if jpg_dir.exists():
        result['image_count'] = len(list(jpg_dir.glob('*.jpg')))
        expected = DATASET_INFO[dataset_name]['images']
        result['images_exist'] = result['image_count'] >= expected * 0.95  # 允许 5% 误差

    # 检查 Ground Truth
    if gnd_file.exists():
        try:
            with open(gnd_file, 'rb') as f:
                gnd = pickle.load(f)
            result['gnd_exist'] = True
            result['gnd_queries'] = len(gnd.get('qimlist', []))
        except Exception as e:
            result['gnd_error'] = str(e)

    result['valid'] = result['images_exist'] and result['gnd_exist']

    return result


def list_datasets():
    """列出所有可用的数据集。"""
    print("\n" + "="*60)
    print("可用的 CIR 评估数据集")
    print("="*60)

    for name, info in DATASET_INFO.items():
        print(f"\n  {name}:")
        print(f"    图像数: {info['images']}")
        print(f"    查询数: {info['queries']}")
        print(f"    来源: {info['source']}")
        print(f"    说明: {info['note']}")

    print("\n使用方法:")
    print(f"  python {Path(__file__).name} --dataset roxford5k")
    print(f"  python {Path(__file__).name} --dataset rparis6k")
    print(f"  python {Path(__file__).name} --all")
    print(f"  python {Path(__file__).name} --verify")


def main():
    parser = argparse.ArgumentParser(
        description='CNN 图像检索数据集下载工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python download_cir_dataset.py --dataset roxford5k              # 下载 ROxford5k
  python download_cir_dataset.py --dataset roxford5k --mirror hf   # 使用 HuggingFace 镜像
  python download_cir_dataset.py --dataset rparis6k               # 下载 RParis6k
  python download_cir_dataset.py --all                            # 下载所有数据集
  python download_cir_dataset.py --verify                         # 验证数据集完整性
  python download_cir_dataset.py --list                           # 列出所有数据集

镜像选项:
  auto     自动选择 (默认，优先 HuggingFace)
  hf       使用 HuggingFace 镜像 (推荐国内用户)
  official 使用官方源

环境变量:
  HF_ENDPOINT  HuggingFace 镜像地址 (如 https://hf-mirror.com)
        """
    )
    parser.add_argument(
        '--dataset', '-d',
        type=str,
        choices=['roxford5k', 'rparis6k'],
        help='要下载的数据集名称'
    )
    parser.add_argument(
        '--all', '-a',
        action='store_true',
        help='下载所有数据集'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default=None,
        help='数据保存目录 (默认: data/test/<dataset>)'
    )
    parser.add_argument(
        '--keep-archive',
        action='store_true',
        help='保留下载的压缩包'
    )
    parser.add_argument(
        '--mirror', '-m',
        type=str,
        choices=['auto', 'hf', 'official'],
        default='auto',
        help='下载源选择: auto(自动), hf(HuggingFace), official(官方源)'
    )
    parser.add_argument(
        '--verify',
        action='store_true',
        help='验证数据集完整性'
    )
    parser.add_argument(
        '--list', '-l',
        action='store_true',
        help='列出所有可用的数据集'
    )

    args = parser.parse_args()

    if args.list:
        list_datasets()
        return

    if args.verify:
        print("\n验证数据集完整性...")
        for dataset in ['roxford5k', 'rparis6k']:
            result = verify_dataset(dataset, args.data_dir)
            status = "✓ 完整" if result['valid'] else "✗ 不完整"
            print(f"\n  {result['name']}: {status}")
            print(f"    路径: {result['path']}")
            print(f"    图像数: {result['image_count']}")
            print(f"    Ground Truth: {'存在' if result['gnd_exist'] else '不存在'}")
        return

    if args.all:
        download_roxford5k(args.data_dir, args.keep_archive, args.mirror)
        download_rparis6k(args.data_dir, args.keep_archive, args.mirror)
        return

    if args.dataset:
        if args.dataset == 'roxford5k':
            download_roxford5k(args.data_dir, args.keep_archive, args.mirror)
        elif args.dataset == 'rparis6k':
            download_rparis6k(args.data_dir, args.keep_archive, args.mirror)
        return

    # 默认显示帮助
    parser.print_help()


if __name__ == '__main__':
    main()