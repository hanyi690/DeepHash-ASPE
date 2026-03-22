"""
快速下载脚本 - 一键下载指定数据集

使用方法:
    python quick_download.py flickr25k
    python quick_download.py iapr_tc12
    python quick_download.py nuswide
    python quick_download.py all
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from data.dataset_downloader import (
    Flickr25KDownloader,
    IAPRTC12Downloader,
    NUSWIDEDownloader,
    download_all_datasets
)
from config.dataset_config import print_download_instructions, check_dataset_exists


def download_flickr25k():
    """下载 Flickr-25K 数据集"""
    print("\n" + "=" * 60)
    print("Flickr-25K 数据集下载")
    print("=" * 60)

    downloader = Flickr25KDownloader('./data/flickr25k')

    # 检查是否已存在
    exists, path, msg = check_dataset_exists('flickr25k')
    if exists:
        print(f"[成功] {msg}")
        if downloader.verify_dataset():
            print("数据集验证通过，可以使用")
            return True

    # 显示下载说明
    print_download_instructions('flickr25k')

    print("\n请手动下载后将文件放置到指定位置")
    print("放置完成后按回车验证...")
    input()

    if downloader.verify_dataset():
        print("\n[成功] Flickr-25K 数据集已就绪")
        return True
    else:
        print("\n[失败] 验证失败，请检查文件路径")
        return False


def download_iapr_tc12():
    """下载 IAPR TC-12 数据集"""
    print("\n" + "=" * 60)
    print("IAPR TC-12 数据集下载")
    print("=" * 60)

    downloader = IAPRTC12Downloader('./data/iapr_tc12')

    # 检查是否已存在
    exists, path, msg = check_dataset_exists('iapr_tc12')
    if exists:
        print(f"[成功] {msg}")
        if downloader.verify_dataset():
            print("数据集验证通过，可以使用")
            return True

    # 显示下载说明
    print_download_instructions('iapr_tc12')

    print("\n请手动下载后将文件放置到指定位置")
    print("放置完成后按回车验证...")
    input()

    if downloader.verify_dataset():
        print("\n[成功] IAPR TC-12 数据集已就绪")
        return True
    else:
        print("\n[失败] 验证失败，请检查文件路径")
        return False


def download_nuswide():
    """下载 NUS-WIDE 数据集"""
    print("\n" + "=" * 60)
    print("NUS-WIDE 数据集下载")
    print("=" * 60)

    downloader = NUSWIDEDownloader('./data/nuswide')

    # 检查是否已存在
    exists, path, msg = check_dataset_exists('nuswide')
    if exists:
        print(f"[成功] {msg}")
        if downloader.verify_dataset():
            print("数据集验证通过，可以使用")
            return True

    # 显示下载说明
    print_download_instructions('nuswide')

    print("\n请手动下载后将文件放置到指定位置")
    print("放置完成后按回车验证...")
    input()

    if downloader.verify_dataset():
        print("\n[成功] NUS-WIDE 数据集已就绪")
        return True
    else:
        print("\n[失败] 验证失败，请检查文件路径")
        return False


def main():
    """主函数"""
    if len(sys.argv) < 2:
        print(__doc__)
        print("\n当前可用数据集:")
        print("  - flickr25k: Flickr-25K (推荐)")
        print("  - iapr_tc12: IAPR TC-12")
        print("  - nuswide: NUS-WIDE")
        print("  - all: 全部下载")
        return

    dataset = sys.argv[1].lower()

    if dataset == 'flickr25k':
        download_flickr25k()
    elif dataset == 'iapr_tc12':
        download_iapr_tc12()
    elif dataset == 'nuswide':
        download_nuswide()
    elif dataset == 'all':
        print("正在下载全部数据集...")
        download_all_datasets()
    else:
        print(f"[错误] 未知数据集：{dataset}")
        print("可用选项：flickr25k, iapr_tc12, nuswide, all")
        sys.exit(1)


if __name__ == "__main__":
    main()
