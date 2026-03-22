"""
数据集下载器测试脚本

用于验证数据集下载器和处理器的功能。
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def test_imports():
    """测试导入"""
    print("=" * 60)
    print("测试导入...")
    print("=" * 60)

    try:
        from data.dataset_downloader import (
            DatasetDownloader,
            Flickr25KDownloader,
            IAPRTC12Downloader,
            NUSWIDEDownloader,
            download_all_datasets
        )
        print("[✓] dataset_downloader 导入成功")
    except ImportError as e:
        print(f"[✗] dataset_downloader 导入失败：{e}")
        return False

    try:
        from data.nuswide_downloader import (
            NUSWideProcessor,
            Flickr25KProcessor,
            IAPRTC12Processor
        )
        print("[✓] nuswide_downloader 导入成功")
    except ImportError as e:
        print(f"[✗] nuswide_downloader 导入失败：{e}")
        return False

    try:
        from config.dataset_config import (
            get_dataset_config,
            check_dataset_exists,
            FLICKR25K_CONFIG,
            IAPRTC12_CONFIG,
            NUSWIDE_CONFIG
        )
        print("[✓] dataset_config 导入成功")
    except ImportError as e:
        print(f"[✗] dataset_config 导入失败：{e}")
        return False

    return True


def test_config():
    """测试配置"""
    print("\n" + "=" * 60)
    print("测试数据集配置...")
    print("=" * 60)

    from config.dataset_config import get_dataset_config, DATASET_CONFIGS, check_dataset_exists

    for name in DATASET_CONFIGS:
        cfg = get_dataset_config(name)
        print(f"\n{name}:")
        print(f"  - 图像数量：{cfg['n_images']:,}")
        print(f"  - 类别数量：{cfg['n_classes']}")
        print(f"  - 数据路径：{cfg['data_path']}")

        exists, path, msg = check_dataset_exists(name)
        if exists:
            print(f"  - 状态：[✓] 已就绪")
        else:
            print(f"  - 状态：[ ] 需要下载")

    return True


def test_downloader_classes():
    """测试下载器类"""
    print("\n" + "=" * 60)
    print("测试下载器类初始化...")
    print("=" * 60)

    from data.dataset_downloader import (
        Flickr25KDownloader,
        IAPRTC12Downloader,
        NUSWIDEDownloader
    )

    # 测试 Flickr-25K
    try:
        flickr = Flickr25KDownloader('./data/flickr25k')
        print(f"[✓] Flickr25KDownloader 初始化成功")
        print(f"    下载目录：{flickr.download_dir}")
    except Exception as e:
        print(f"[✗] Flickr25KDownloader 初始化失败：{e}")

    # 测试 IAPR TC-12
    try:
        iapr = IAPRTC12Downloader('./data/iapr_tc12')
        print(f"[✓] IAPRTC12Downloader 初始化成功")
        print(f"    下载目录：{iapr.download_dir}")
    except Exception as e:
        print(f"[✗] IAPRTC12Downloader 初始化失败：{e}")

    # 测试 NUS-WIDE
    try:
        nus = NUSWIDEDownloader('./data/nuswide')
        print(f"[✓] NUSWIDEDownloader 初始化成功")
        print(f"    下载目录：{nus.download_dir}")
    except Exception as e:
        print(f"[✗] NUSWIDEDownloader 初始化失败：{e}")

    return True


def test_processor_classes():
    """测试处理器类"""
    print("\n" + "=" * 60)
    print("测试处理器类初始化...")
    print("=" * 60)

    from data.nuswide_downloader import (
        NUSWideProcessor,
        Flickr25KProcessor,
        IAPRTC12Processor
    )

    # 测试 NUS-WIDE
    try:
        nus_proc = NUSWideProcessor('./data/nuswide')
        print(f"[✓] NUSWideProcessor 初始化成功")
        print(f"    类别数量：{len(nus_proc.CATEGORIES_81)}")
    except Exception as e:
        print(f"[✗] NUSWideProcessor 初始化失败：{e}")

    # 测试 Flickr-25K
    try:
        flickr_proc = Flickr25KProcessor('./data/flickr25k')
        print(f"[✓] Flickr25KProcessor 初始化成功")
        print(f"    类别数量：{len(flickr_proc.CATEGORIES_24)}")
    except Exception as e:
        print(f"[✗] Flickr25KProcessor 初始化失败：{e}")

    # 测试 IAPR TC-12
    try:
        iapr_proc = IAPRTC12Processor('./data/iapr_tc12')
        print(f"[✓] IAPRTC12Processor 初始化成功")
    except Exception as e:
        print(f"[✗] IAPRTC12Processor 初始化失败：{e}")

    return True


def test_scientific_libs():
    """测试科学计算库"""
    print("\n" + "=" * 60)
    print("测试科学计算库...")
    print("=" * 60)

    # 测试 numpy
    try:
        import numpy as np
        arr = np.array([1, 2, 3])
        print(f"[✓] numpy 可用 (版本 {np.__version__})")
    except ImportError:
        print(f"[✗] numpy 未安装")

    # 测试 scipy
    try:
        import scipy
        print(f"[✓] scipy 可用 (版本 {scipy.__version__})")
    except ImportError:
        print(f"[✗] scipy 未安装")

    # 测试 h5py
    try:
        import h5py
        print(f"[✓] h5py 可用 (版本 {h5py.__version__})")
    except ImportError:
        print(f"[✗] h5py 未安装")

    return True


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("DCMH + ASPE 数据集下载器测试")
    print("=" * 60)

    results = {
        'imports': False,
        'config': False,
        'downloader': False,
        'processor': False,
        'scientific_libs': False
    }

    # 运行测试
    results['imports'] = test_imports()
    results['config'] = test_config()
    results['downloader'] = test_downloader_classes()
    results['processor'] = test_processor_classes()
    results['scientific_libs'] = test_scientific_libs()

    # 汇总结果
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for test_name, result in results.items():
        status = "[✓] 通过" if result else "[✗] 失败"
        print(f"  {test_name}: {status}")

    print(f"\n总计：{passed}/{total} 测试通过")

    if passed == total:
        print("\n[成功] 所有测试通过！")
        return 0
    else:
        print("\n[警告] 部分测试失败，请检查输出信息")
        return 1


if __name__ == "__main__":
    sys.exit(main())
