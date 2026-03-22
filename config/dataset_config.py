"""
数据集配置文件

用于配置 DCMH + ASPE 系统测试所需的数据集路径和参数。
"""

from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent

# 数据根目录
DATA_ROOT = PROJECT_ROOT / 'data'


# =============================================================================
# Flickr-25K 数据集配置
# =============================================================================

FLICKR25K_CONFIG = {
    # 数据路径
    'data_path': DATA_ROOT / 'flickr25k' / 'FLICKR-25K.mat',

    # 数据集参数
    'n_classes': 24,
    'n_images': 25015,

    # 数据集划分（参考实现）
    'training_size': 10000,
    'query_size': 2000,
    'database_size': 18015,

    # 类别名称
    'categories': [
        'airplane', 'bird', 'boat', 'building', 'car', 'cat',
        'dog', 'field', 'flower', 'food', 'grass', 'horse',
        'mountain', 'person', 'sheep', 'sky', 'street', 'table',
        'tower', 'tree', 'water', 'wave', 'window', 'woman'
    ],

    # 下载说明
    'download_info': {
        'source': '百度网盘',
        'url': 'https://pan.baidu.com/s/1X5BTyux524aUyqHpFGPPlA',
        'password': 'eico',
        'note': '下载 FLICKR-25K.mat 并放置到 data/flickr25k/ 目录'
    }
}


# =============================================================================
# IAPR TC-12 数据集配置
# =============================================================================

IAPRTC12_CONFIG = {
    # 数据路径
    'data_path': DATA_ROOT / 'iapr_tc12' / 'IAPR-TC12.mat',

    # 数据集参数
    'n_classes': 255,
    'n_images': 22127,

    # 数据集划分
    'training_size': 10000,
    'query_size': 2000,
    'database_size': 18127,

    # 类别名称（动态加载）
    'categories': [],

    # 下载说明
    'download_info': {
        'source': 'ImageCLEF',
        'url': 'http://www.imageclef.org/photodata',
        'note': '需要注册并申请下载'
    }
}


# =============================================================================
# NUS-WIDE 数据集配置
# =============================================================================

NUSWIDE_CONFIG = {
    # 数据路径
    'data_path': DATA_ROOT / 'nuswide' / 'NUS-WIDE.mat',

    # 数据集参数
    'n_classes': 81,
    'n_images': 269648,  # 原始数量

    # 数据集划分
    'training_size': 105000,
    'query_size': 2100,
    'database_size': 165000,

    # 类别名称
    'categories': [
        'airplane', 'animal', 'bear', 'bird', 'boats', 'book',
        'bridge', 'building', 'bus', 'car', 'cat', 'chair',
        'clouds', 'computer', 'cow', 'crosswalk', 'desert', 'dog',
        'field', 'fire hydrant', 'flag', 'flower', 'food', 'forest',
        'furniture', 'grass', 'harbor', 'house', 'jetty', 'kid',
        'lake', 'lawn', 'man', 'meeting room', 'mountain', 'pavement',
        'person', 'platform', 'plant', 'pot', 'racing', 'refrigerator',
        'river', 'road', 'rock', 'sailing boat', 'sand', 'sea',
        'sheep', 'sign', 'sky', 'snow', 'stairs', 'street', 'sun',
        'swimming pool', 'table', 'tower', 'track', 'train', 'tree',
        'truck', 'umbrella', 'vehicle', 'wall', 'water', 'waterfall',
        'wave', 'wetland', 'window', 'woman', 'zebra'
    ],

    # 下载说明
    'download_info': {
        'source': 'NUS',
        'url': 'http://lms.comp.nus.edu.sg/research/nuswide.shtml',
        'note': '下载 ImageList.txt 和 Groundtruth 文件'
    }
}


# =============================================================================
# 默认数据集配置
# =============================================================================

# 默认使用 Flickr-25K（参考实现原生支持）
DEFAULT_DATASET = 'flickr25k'

DATASET_CONFIGS = {
    'flickr25k': FLICKR25K_CONFIG,
    'iapr_tc12': IAPRTC12_CONFIG,
    'nuswide': NUSWIDE_CONFIG
}


def get_dataset_config(dataset_name: str = None):
    """
    获取数据集配置

    参数:
        dataset_name: 数据集名称 ('flickr25k' | 'iapr_tc12' | 'nuswide')

    返回:
        数据集配置字典
    """
    if dataset_name is None:
        dataset_name = DEFAULT_DATASET

    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"未知数据集：{dataset_name}")

    return DATASET_CONFIGS[dataset_name]


def check_dataset_exists(dataset_name: str = None) -> tuple:
    """
    检查数据集是否存在

    参数:
        dataset_name: 数据集名称

    返回:
        (exists: bool, path: Path, message: str)
    """
    config = get_dataset_config(dataset_name)
    data_path = Path(config['data_path'])

    if data_path.exists():
        return True, data_path, f"数据集已就绪：{data_path}"
    else:
        return False, data_path, f"数据集不存在，请下载：{data_path}"


def print_download_instructions(dataset_name: str = None):
    """
    打印数据集下载说明

    参数:
        dataset_name: 数据集名称
    """
    config = get_dataset_config(dataset_name)
    download_info = config['download_info']

    print(f"\n{'='*60}")
    print(f"{dataset_name or DEFAULT_DATASET} 数据集下载说明")
    print(f"{'='*60}")
    print(f"来源：{download_info.get('source', 'Unknown')}")
    print(f"URL: {download_info.get('url', 'N/A')}")
    if 'password' in download_info:
        print(f"密码：{download_info['password']}")
    print(f"说明：{download_info.get('note', '')}")
    print(f"目标路径：{config['data_path']}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    # 打印所有数据集的下载说明
    for name in DATASET_CONFIGS.keys():
        print_download_instructions(name)
