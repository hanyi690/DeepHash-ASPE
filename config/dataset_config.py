"""
数据集配置文件

用于配置 DCMH + ASPE 系统测试所需的数据集路径和参数。
支持：
- DCMH 跨模态检索数据集（Flickr25K、IAPR TC-12、NUS-WIDE）
- CIR 图像检索数据集（ROxford5k、RParis6k）
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

    # 类别名称（按 LAll 索引顺序，与 mirflickr25k_annotations_v080 标注文件对应）
    'categories': [
        'animals', 'baby', 'bird', 'car', 'clouds', 'dog',
        'female', 'flower', 'food', 'indoor', 'lake', 'male',
        'night', 'people', 'plant_life', 'portrait', 'river', 'sea',
        'sky', 'structures', 'sunset', 'transport', 'tree', 'water'
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
# CIR 图像检索数据集配置
# =============================================================================

# ROxford5k 数据集（Revisited Oxford5k）
ROXFORD5K_CONFIG = {
    # 数据路径
    'images_dir': DATA_ROOT / 'test' / 'roxford5k' / 'jpg',
    'gnd_path': DATA_ROOT / 'test' / 'roxford5k' / 'gnd_roxford5k.pkl',

    # 数据集参数
    'n_images': 5062,
    'n_queries': 70,

    # 数据集信息
    'source': 'Oxford Buildings',
    'note': 'Revisited 版本，包含更准确的标注',

    # 下载说明
    'download_info': {
        'source': 'Oxford VGG / CNN Image Retrieval',
        'images_url': 'https://www.robots.ox.ac.uk/~vgg/data/oxbuildings/oxbuild_images-v1.tgz',
        'gnd_url': 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/test/roxford5k/gnd_roxford5k.pkl',
        'script': 'python scripts/download_cir_dataset.py --dataset roxford5k'
    }
}

# RParis6k 数据集（Revisited Paris6k）
RPARIS6K_CONFIG = {
    # 数据路径
    'images_dir': DATA_ROOT / 'test' / 'rparis6k' / 'jpg',
    'gnd_path': DATA_ROOT / 'test' / 'rparis6k' / 'gnd_rparis6k.pkl',

    # 数据集参数
    'n_images': 6392,
    'n_queries': 70,

    # 数据集信息
    'source': 'Paris Buildings',
    'note': 'Revisited 版本，包含更准确的标注',

    # 下载说明
    'download_info': {
        'source': 'Oxford VGG / CNN Image Retrieval',
        'images_url': 'https://www.robots.ox.ac.uk/~vgg/data/parisbuildings/',
        'gnd_url': 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/test/rparis6k/gnd_rparis6k.pkl',
        'script': 'python scripts/download_cir_dataset.py --dataset rparis6k'
    }
}


# =============================================================================
# CIR 模型配置
# =============================================================================

CIR_MODEL_CONFIG = {
    # 推荐模型（最高性能）
    'gl18-resnet101-gem-w': {
        'path': DATA_ROOT / 'networks' / 'gl18-tl-resnet101-gem-w-a4d43db.pth',
        'desc': 'GL18 ResNet101-GeM-W（推荐，最高性能）',
        'performance': {'ROxf(M)': 67.3, 'RPar(M)': 80.6},
        'download_script': 'python scripts/download_cir_model.py --model gl18-resnet101-gem-w'
    },
    # SfM-120k 模型
    'sfm120k-resnet101-gem': {
        'path': DATA_ROOT / 'networks' / 'retrievalSfM120k-resnet101-gem-b80fb85.pth',
        'desc': 'SfM-120k ResNet101-GeM',
        'performance': {'ROxf(M)': 65.4, 'RPar(M)': 76.7},
        'download_script': 'python scripts/download_cir_model.py --model sfm120k-resnet101-gem'
    },
    # ImageNet 特征模型（off-the-shelf 模式）
    'resnet50-caffe': {
        'path': DATA_ROOT / 'networks' / 'imagenet-caffe-resnet50-features-ac468af.pth',
        'desc': 'ResNet50 ImageNet Caffe 特征',
        'download_script': 'python scripts/download_cir_model.py --model resnet50-caffe'
    },
    # 白化权重
    'resnet101-gem-whiten': {
        'path': DATA_ROOT / 'whiten' / 'retrieval-SfM-120k-resnet101-gem-whiten-22ab0c1.pth',
        'desc': 'ResNet101-GeM 白化权重',
        'download_script': 'python scripts/download_cir_model.py --model resnet101-caffe'
    }
}


# =============================================================================
# 默认数据集配置
# =============================================================================

# 默认使用 Flickr-25K（参考实现原生支持）
DEFAULT_DATASET = 'flickr25k'

# DCMH 跨模态检索数据集
DATASET_CONFIGS = {
    'flickr25k': FLICKR25K_CONFIG,
    'iapr_tc12': IAPRTC12_CONFIG,
    'nuswide': NUSWIDE_CONFIG
}

# CIR 图像检索数据集
CIR_DATASET_CONFIGS = {
    'roxford5k': ROXFORD5K_CONFIG,
    'rparis6k': RPARIS6K_CONFIG
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


def get_cir_dataset_config(dataset_name: str):
    """
    获取 CIR 数据集配置

    参数:
        dataset_name: 数据集名称 ('roxford5k' | 'rparis6k')

    返回:
        数据集配置字典
    """
    if dataset_name not in CIR_DATASET_CONFIGS:
        raise ValueError(f"未知 CIR 数据集：{dataset_name}，可用：{list(CIR_DATASET_CONFIGS.keys())}")

    return CIR_DATASET_CONFIGS[dataset_name]


def get_cir_model_config(model_name: str):
    """
    获取 CIR 模型配置

    参数:
        model_name: 模型名称

    返回:
        模型配置字典
    """
    if model_name not in CIR_MODEL_CONFIG:
        raise ValueError(f"未知 CIR 模型：{model_name}，可用：{list(CIR_MODEL_CONFIG.keys())}")

    return CIR_MODEL_CONFIG[model_name]


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


def check_cir_dataset_exists(dataset_name: str) -> tuple:
    """
    检查 CIR 数据集是否存在

    参数:
        dataset_name: 数据集名称 ('roxford5k' | 'rparis6k')

    返回:
        (exists: bool, path: Path, message: str)
    """
    config = get_cir_dataset_config(dataset_name)
    images_dir = Path(config['images_dir'])
    gnd_path = Path(config['gnd_path'])

    images_exist = images_dir.exists() and len(list(images_dir.glob('*.jpg'))) > 0
    gnd_exist = gnd_path.exists()

    if images_exist and gnd_exist:
        return True, images_dir.parent, f"CIR 数据集已就绪：{images_dir.parent}"
    else:
        missing = []
        if not images_exist:
            missing.append(f"图像目录 {images_dir}")
        if not gnd_exist:
            missing.append(f"Ground Truth {gnd_path}")
        return False, images_dir.parent, f"CIR 数据集不完整，缺少：{', '.join(missing)}"


def check_cir_model_exists(model_name: str) -> tuple:
    """
    检查 CIR 模型是否存在

    参数:
        model_name: 模型名称

    返回:
        (exists: bool, path: Path, message: str)
    """
    config = get_cir_model_config(model_name)
    model_path = Path(config['path'])

    if model_path.exists():
        return True, model_path, f"模型已就绪：{model_path}"
    else:
        return False, model_path, f"模型不存在，请运行：{config['download_script']}"


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


def print_cir_download_instructions(dataset_name: str = None):
    """
    打印 CIR 数据集下载说明

    参数:
        dataset_name: 数据集名称，为 None 时打印所有
    """
    if dataset_name:
        datasets = {dataset_name: CIR_DATASET_CONFIGS[dataset_name]}
    else:
        datasets = CIR_DATASET_CONFIGS

    for name, config in datasets.items():
        download_info = config['download_info']
        print(f"\n{'='*60}")
        print(f"{name} CIR 数据集下载说明")
        print(f"{'='*60}")
        print(f"来源：{download_info.get('source', 'Unknown')}")
        print(f"图像 URL: {download_info.get('images_url', 'N/A')}")
        print(f"Ground Truth URL: {download_info.get('gnd_url', 'N/A')}")
        print(f"图像数：{config['n_images']}")
        print(f"查询数：{config['n_queries']}")
        print(f"下载命令：{download_info.get('script', '')}")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    # 打印所有数据集的下载说明
    print("\n" + "="*60)
    print("DCMH 跨模态检索数据集")
    print("="*60)
    for name in DATASET_CONFIGS.keys():
        print_download_instructions(name)

    print("\n" + "="*60)
    print("CIR 图像检索数据集")
    print("="*60)
    print_cir_download_instructions()

    print("\n" + "="*60)
    print("CIR 预训练模型")
    print("="*60)
    for name, config in CIR_MODEL_CONFIG.items():
        exists, path, msg = check_cir_model_exists(name)
        status = "✓" if exists else "✗"
        print(f"\n  {status} {name}: {config['desc']}")
        print(f"      路径: {path}")
        if not exists:
            print(f"      下载: {config['download_script']}")
