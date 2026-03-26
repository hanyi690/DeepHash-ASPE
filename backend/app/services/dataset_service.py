"""
数据集服务：加载和管理多个数据集

支持的数据集：
- Flickr25K: 跨模态检索数据集（直接从 .mat 文件读取）
- NUS-WIDE: 跨模态检索数据集（.mat 格式）

简化设计：
- 直接使用 .mat 文件中的 YAll 和 images 数据
- 使用 tag_mapping.npy 映射 YAll 列索引到标签名
- 仅在需要显示图像时，使用 clean_id 映射到原始图像文件
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from typing import Optional, Tuple, Dict, Any, List
from pathlib import Path
from PIL import Image
import sys
import logging
import h5py

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)

# 数据集配置
DATASET_CONFIGS = {
    'flickr25k': {
        'mat_path': 'data/FLICKR-25K.mat',              # .mat 文件路径
        'images_dir': 'data/dcmh/flickr25k/mirflickr',  # 原始图像目录（用于显示）
        'tag_mapping_path': 'data/dcmh/flickr25k/tag_mapping.npy',  # YAll 列到标签名映射
        'clean_id_path': 'data/dcmh/flickr25k/clean_id.flickr25k.mat',  # 清洗ID映射
        'query_size': 2000,
        'training_size': 10000,
        'database_size': 18015,
        'y_dim': 1386,                                  # YAll 标签维度
        'l_dim': 24,                                    # LAll 类别维度
        'display_name': 'Flickr25K',
    },
    'nuswide': {
        'mat_path': 'data/NUS-WIDE.mat',
        'query_size': 2100,
        'training_size': 10500,
        'database_size': 193734,
        'y_dim': 81,
        'l_dim': 81,
        'display_name': 'NUS-WIDE',
    }
}

# 支持的数据集列表
DCMH_DATASETS = list(DATASET_CONFIGS.keys())


class MatDatasetService:
    """
    基于 .mat 文件的数据集服务。

    直接从 .mat 文件读取数据，避免复杂的原始文件映射。
    """

    def __init__(self, dataset_name: str = 'flickr25k'):
        """
        初始化服务。

        参数：
            dataset_name: 数据集名称
        """
        if dataset_name not in DATASET_CONFIGS:
            raise ValueError(f"不支持的数据集：{dataset_name}")

        self.dataset_name = dataset_name
        self.config = DATASET_CONFIGS[dataset_name]
        self.mat_path = PROJECT_ROOT / self.config['mat_path']

        # h5py 文件句柄
        self._h5_file: Optional[h5py.File] = None

        # 数据缓存
        self._yall: Optional[np.ndarray] = None
        self._lall: Optional[np.ndarray] = None
        self._images: Optional[np.ndarray] = None

        # 标签映射
        self._tag_mapping: Optional[List[str]] = None

        # clean_id 映射（用于图像显示）
        self._clean_id: Optional[np.ndarray] = None

        # 数据划分
        self._data_loaded = False
        self._train_indices = None
        self._query_indices = None
        self._retrieval_indices = None

    def _load_h5_file(self):
        """延迟加载 h5 文件。"""
        if self._h5_file is None:
            if not self.mat_path.exists():
                raise FileNotFoundError(f"数据文件不存在：{self.mat_path}")
            self._h5_file = h5py.File(self.mat_path, 'r')
            logger.info(f"[MatDatasetService] 加载数据文件：{self.mat_path}")

    def _load_tag_mapping(self):
        """加载 YAll 列到标签名的映射。"""
        if self._tag_mapping is not None:
            return

        tag_mapping_path = self.config.get('tag_mapping_path')
        if tag_mapping_path:
            path = PROJECT_ROOT / tag_mapping_path
            if path.exists():
                self._tag_mapping = list(np.load(path, allow_pickle=True))
                logger.info(f"[MatDatasetService] 加载标签映射：{len(self._tag_mapping)} 个")

    def _load_clean_id(self):
        """加载 clean_id 映射。"""
        if self._clean_id is not None:
            return

        clean_id_path = self.config.get('clean_id_path')
        if clean_id_path:
            path = PROJECT_ROOT / clean_id_path
            if path.exists():
                from scipy.io import loadmat
                self._clean_id = loadmat(path)['clean_id'].flatten()
                logger.info(f"[MatDatasetService] 加载 clean_id：{len(self._clean_id)} 个")

    @property
    def h5_file(self) -> h5py.File:
        """获取 h5 文件句柄。"""
        self._load_h5_file()
        return self._h5_file

    @property
    def y_dim(self) -> int:
        """标签向量维度。"""
        return self.config['y_dim']

    @property
    def l_dim(self) -> int:
        """类别维度。"""
        return self.config['l_dim']

    @property
    def query_size(self) -> int:
        return self.config['query_size']

    @property
    def training_size(self) -> int:
        return self.config['training_size']

    @property
    def database_size(self) -> int:
        return self.config['database_size']

    def load_data(self):
        """加载数据划分索引。"""
        if self._data_loaded:
            return

        self._train_indices = np.arange(self.query_size,
                                        self.query_size + self.training_size)
        self._query_indices = np.arange(0, self.query_size)
        self._retrieval_indices = np.arange(self.query_size,
                                            self.query_size + self.database_size)

        self._data_loaded = True
        logger.info(f"[MatDatasetService] 数据划分：查询 {len(self._query_indices)}，"
                   f"训练 {len(self._train_indices)}，检索 {len(self._retrieval_indices)}")

    def get_data_split_indices(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """获取数据划分索引。"""
        if not self._data_loaded:
            self.load_data()
        return self._train_indices, self._query_indices, self._retrieval_indices

    def get_yall(self, indices: Optional[np.ndarray] = None) -> np.ndarray:
        """
        获取 YAll 标签向量。

        参数：
            indices: 索引数组（可选）

        返回：
            YAll 数组 [N, y_dim]
        """
        if indices is not None:
            data = self.h5_file['YAll'][indices]
        else:
            data = self.h5_file['YAll'][:]

        # 处理稀疏矩阵
        if hasattr(data, 'toarray'):
            data = data.toarray()
        return data.astype(np.float32)

    def get_lall(self, indices: Optional[np.ndarray] = None) -> np.ndarray:
        """
        获取 LAll 类别标签。

        参数：
            indices: 索引数组（可选）

        返回：
            LAll 数组 [N, l_dim]
        """
        if indices is not None:
            data = self.h5_file['LAll'][indices]
        else:
            data = self.h5_file['LAll'][:]

        if hasattr(data, 'toarray'):
            data = data.toarray()
        return data.astype(np.float32)

    def get_images(self, indices: Optional[np.ndarray] = None) -> np.ndarray:
        """
        获取预处理后的图像数据。

        参数：
            indices: 索引数组（可选）

        返回：
            图像数组 [N, 3, 224, 224]
        """
        if indices is not None:
            return self.h5_file['images'][indices].astype(np.float32)
        return self.h5_file['images'][:].astype(np.float32)

    def get_tags(self, indices: Optional[np.ndarray] = None) -> np.ndarray:
        """获取文本标签（兼容旧接口）。"""
        return self.get_yall(indices)

    def get_labels(self, indices: Optional[np.ndarray] = None) -> np.ndarray:
        """获取标签（兼容旧接口）。"""
        return self.get_lall(indices)

    def get_tag_names(self) -> List[str]:
        """
        获取标签名列表。

        返回 YAll 列索引对应的标签名列表。
        """
        self._load_tag_mapping()
        if self._tag_mapping is not None:
            return self._tag_mapping
        return [f'tag_{i}' for i in range(self.y_dim)]

    def get_tag_names_from_yall_indices(self, yall_indices: List[int]) -> List[str]:
        """
        根据 YAll 索引获取标签名。

        参数：
            yall_indices: YAll 列索引列表

        返回：
            标签名称列表
        """
        tag_names = self.get_tag_names()
        return [tag_names[i] if 0 <= i < len(tag_names) else f'tag_{i}'
                for i in yall_indices]

    def get_category_names(self) -> List[str]:
        """
        获取 LAll 类别名称列表。

        返回 LAll 列索引对应的类别名称列表。
        """
        from config.dataset_config import get_dataset_config
        try:
            config = get_dataset_config(self.dataset_name)
            categories = config.get('categories', [])
            if categories:
                return categories
        except Exception:
            pass
        return [f'category_{i}' for i in range(self.l_dim)]

    def get_category_names_from_lall_indices(self, lall_indices: List[int]) -> List[str]:
        """
        根据 LAll 索引获取类别名称。

        参数：
            lall_indices: LAll 列索引列表

        返回：
            类别名称列表
        """
        category_names = self.get_category_names()
        return [category_names[i] if 0 <= i < len(category_names) else f'category_{i}'
                for i in lall_indices]

    def get_original_image_id(self, idx: int) -> int:
        """
        将 .mat 文件索引转换为原始图像 ID。

        用于生成图像 URL：im{original_id + 1}.jpg

        参数：
            idx: .mat 文件中的行索引

        返回：
            原始图像 ID（0-based）
        """
        self._load_clean_id()
        if self._clean_id is not None and 0 <= idx < len(self._clean_id):
            return int(self._clean_id[idx])
        return idx

    def get_image_count(self) -> int:
        """获取图像总数。"""
        return self.h5_file['images'].shape[0]

    def create_image_dataloader(self,
                                indices: np.ndarray,
                                batch_size: int = 64,
                                shuffle: bool = False,
                                num_workers: int = 0) -> DataLoader:
        """
        创建图像 DataLoader。

        直接从 .mat 文件读取预处理图像。
        """
        dataset = MatImageDataset(self.mat_path, indices)
        return DataLoader(dataset, batch_size=batch_size,
                         shuffle=shuffle, num_workers=num_workers)

    def create_text_dataset(self, indices: np.ndarray, normalize: bool = False):
        """
        创建文本数据集。

        直接从 .mat 文件读取 YAll。

        参数：
            indices: 样本索引数组
            normalize: 是否零均值归一化（默认 False，与训练时保持一致）
        """
        return MatTextDataset(self.mat_path, indices, normalize=normalize)

    def get_status(self) -> Dict[str, Any]:
        """获取服务状态。"""
        return {
            "dataset_name": self.dataset_name,
            "display_name": self.config['display_name'],
            "mat_path": str(self.mat_path),
            "data_loaded": self._data_loaded,
            "mat_exists": self.mat_path.exists(),
            "query_size": self.query_size,
            "training_size": self.training_size,
            "database_size": self.database_size,
            "y_dim": self.y_dim,
            "l_dim": self.l_dim
        }

    def close(self):
        """关闭资源。"""
        if self._h5_file is not None:
            self._h5_file.close()
            self._h5_file = None


class MatImageDataset(Dataset):
    """
    从 .mat 文件读取图像数据的数据集。

    用于生成图像哈希码。
    """

    def __init__(self, mat_path: Path, indices: np.ndarray):
        """
        初始化数据集。

        参数：
            mat_path: .mat 文件路径
            indices: 样本索引数组
        """
        self.mat_path = Path(mat_path)
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        actual_idx = self.indices[idx]

        with h5py.File(self.mat_path, 'r') as f:
            img = f['images'][actual_idx].astype(np.float32)

        # 转换为 tensor
        img_tensor = torch.from_numpy(img)

        # 返回 local idx（用于 F_buffer 索引）
        return img_tensor, idx


class MatTextDataset(Dataset):
    """
    从 .mat 文件读取文本标签的数据集。

    用于生成文本哈希码。
    """

    def __init__(self, mat_path: Path, indices: np.ndarray, normalize: bool = False):
        """
        初始化数据集。

        参数：
            mat_path: .mat 文件路径
            indices: 样本索引数组
            normalize: 是否零均值归一化（默认 False，与训练时保持一致）
        """
        self.mat_path = Path(mat_path)
        self.indices = indices
        self.normalize = normalize
        self._mean: Optional[np.ndarray] = None

    def __len__(self):
        return len(self.indices)

    def _compute_mean(self) -> np.ndarray:
        """计算标签均值。"""
        if self._mean is None:
            with h5py.File(self.mat_path, 'r') as f:
                yall = f['YAll'][self.indices]
                if hasattr(yall, 'toarray'):
                    yall = yall.toarray()
                self._mean = yall.mean(axis=0)
        return self._mean

    def __getitem__(self, idx):
        actual_idx = self.indices[idx]

        with h5py.File(self.mat_path, 'r') as f:
            yall = f['YAll'][actual_idx]
            if hasattr(yall, 'toarray'):
                yall = yall.toarray()
            tags = yall.flatten().astype(np.float32)

        # 零均值归一化
        if self.normalize:
            mean = self._compute_mean()
            tags = tags - mean

        # 转换为 tensor [1, y_dim, 1]
        tags_tensor = torch.from_numpy(tags).float().unsqueeze(0).unsqueeze(-1)

        # 未归一化的标签（用于命中判断）
        with h5py.File(self.mat_path, 'r') as f:
            yall = f['YAll'][actual_idx]
            if hasattr(yall, 'toarray'):
                yall = yall.toarray()
            tags_raw = torch.from_numpy(yall.flatten().astype(np.float32))

        return tags_tensor, tags_raw, idx


# 兼容旧接口
DatasetService = MatDatasetService


# 全局服务实例缓存
_dataset_services: Dict[str, MatDatasetService] = {}


def get_dataset_service(dataset_name: str = 'flickr25k', **kwargs) -> MatDatasetService:
    """
    获取或创建数据集服务实例。

    参数：
        dataset_name: 数据集名称

    返回：
        MatDatasetService 实例
    """
    if dataset_name not in _dataset_services:
        _dataset_services[dataset_name] = MatDatasetService(dataset_name=dataset_name)
    return _dataset_services[dataset_name]


def get_y_dim_for_dataset(dataset_name: str) -> int:
    """获取指定数据集的标签维度。"""
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"不支持的数据集：{dataset_name}")
    return DATASET_CONFIGS[dataset_name]['y_dim']