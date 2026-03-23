"""
数据集服务：加载和管理 Flickr25K 数据集

提供：
- 加载 Flickr25K 数据
- 按需读取图像和标签
- 数据划分 (train/query/retrieval)
"""

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader
from typing import Optional, Tuple, Dict, Any
from pathlib import Path
import sys

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from training.dcmh_dataset import DCMHImageDataset, DCMHTextDataset

# 默认数据路径
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "FLICKR-25K.mat"


class DatasetService:
    """
    数据集服务。

    提供：
    - 加载 Flickr25K 数据
    - 数据划分
    - 创建 DataLoader
    """

    # Flickr25K 数据划分
    QUERY_SIZE = 2000
    TRAINING_SIZE = 10000
    DATABASE_SIZE = 18015

    def __init__(self, data_path: Optional[str] = None):
        """
        初始化数据集服务。

        参数：
            data_path: FLICKR-25K.mat 文件路径
        """
        self.data_path = data_path or str(DEFAULT_DATA_PATH)
        self._h5_file = None
        self._data_loaded = False

        # 缓存数据
        self._train_indices = None
        self._query_indices = None
        self._retrieval_indices = None

    @property
    def h5_file(self):
        """延迟加载 h5 文件。"""
        if self._h5_file is None:
            if not Path(self.data_path).exists():
                raise FileNotFoundError(f"数据文件不存在：{self.data_path}")
            self._h5_file = h5py.File(self.data_path, 'r')
        return self._h5_file

    def load_data(self):
        """加载数据集元信息。"""
        if self._data_loaded:
            return

        # 划分索引
        self._train_indices = np.arange(self.QUERY_SIZE,
                                         self.QUERY_SIZE + self.TRAINING_SIZE)
        self._query_indices = np.arange(0, self.QUERY_SIZE)
        self._retrieval_indices = np.arange(self.QUERY_SIZE,
                                            self.QUERY_SIZE + self.DATABASE_SIZE)

        self._data_loaded = True
        print(f"数据集已加载：{self.data_path}")
        print(f"  - 训练集大小：{len(self._train_indices)}")
        print(f"  - 查询集大小：{len(self._query_indices)}")
        print(f"  - 检索库大小：{len(self._retrieval_indices)}")

    def get_data_split_indices(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        获取数据划分索引。

        返回：
            (train_indices, query_indices, retrieval_indices)
        """
        if not self._data_loaded:
            self.load_data()
        return self._train_indices, self._query_indices, self._retrieval_indices

    def get_labels(self, indices: Optional[np.ndarray] = None) -> np.ndarray:
        """
        获取标签。

        参数：
            indices: 可选的索引数组

        返回：
            标签数组 [N, L]
        """
        if indices is not None:
            return self.h5_file['LAll'][indices]
        return self.h5_file['LAll'][:]

    def get_tags(self, indices: Optional[np.ndarray] = None) -> np.ndarray:
        """
        获取文本标签（multi-hot 向量）。

        参数：
            indices: 可选的索引数组

        返回：
            标签数组 [N, y_dim]
        """
        if indices is not None:
            return self.h5_file['YAll'][indices]
        return self.h5_file['YAll'][:]

    def get_image_count(self) -> int:
        """获取图像总数。"""
        return len(self.h5_file['images'])

    def create_image_dataloader(self,
                                indices: np.ndarray,
                                batch_size: int = 64,
                                shuffle: bool = False,
                                num_workers: int = 0) -> DataLoader:
        """
        创建图像 DataLoader。

        参数：
            indices: 图像索引数组
            batch_size: 批次大小
            shuffle: 是否打乱
            num_workers: 工作线程数

        返回：
            DataLoader 实例
        """
        from torchvision import transforms

        # 图像变换
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

        dataset = DCMHImageDataset(self.data_path, indices, transform)
        return DataLoader(dataset, batch_size=batch_size,
                         shuffle=shuffle, num_workers=num_workers)

    def create_text_dataset(self, indices: np.ndarray) -> DCMHTextDataset:
        """
        创建文本数据集。

        参数：
            indices: 文本索引数组

        返回：
            DCMHTextDataset 实例
        """
        return DCMHTextDataset(self.data_path, indices)

    def get_status(self) -> Dict[str, Any]:
        """获取服务状态。"""
        return {
            "data_path": self.data_path,
            "data_loaded": self._data_loaded,
            "data_exists": Path(self.data_path).exists(),
            "query_size": self.QUERY_SIZE,
            "training_size": self.TRAINING_SIZE,
            "database_size": self.DATABASE_SIZE
        }

    def close(self):
        """关闭 h5 文件。"""
        if self._h5_file is not None:
            self._h5_file.close()
            self._h5_file = None


# 全局服务实例
_dataset_service: Optional[DatasetService] = None


def get_dataset_service(**kwargs) -> DatasetService:
    """获取或创建数据集服务单例。"""
    global _dataset_service
    if _dataset_service is None:
        _dataset_service = DatasetService(**kwargs)
    return _dataset_service