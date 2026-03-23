"""
DCMH 数据集类

提供按需加载图像的 Dataset 类，大幅降低内存占用。
"""

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import io


class DCMHImageDataset(Dataset):
    """
    DCMH 图像数据集（按需加载）。

    不从 h5py 加载所有图像到内存，而是在 __getitem__ 时按需读取单张图像。
    内存占用：从 10GB+ 降至 < 100MB
    """

    def __init__(self, h5_path, indices=None, transform=None):
        """
        初始化数据集。

        参数：
            h5_path: FLICKR-25K.mat 文件路径
            indices: 要使用的样本索引（用于 train/query/retrieval 划分）
            transform: 图像变换（可选）
        """
        self.h5_path = h5_path
        self.indices = indices
        self.transform = transform
        self._h5_file = None

    @property
    def h5_file(self):
        """延迟打开 h5 文件（每个进程一个句柄）"""
        if self._h5_file is None:
            self._h5_file = h5py.File(self.h5_path, 'r')
        return self._h5_file

    def __len__(self):
        if self.indices is not None:
            return len(self.indices)
        return len(self.h5_file['images'])

    def __getitem__(self, idx):
        # 获取实际索引
        if self.indices is not None:
            actual_idx = self.indices[idx]
        else:
            actual_idx = idx

        # 读取图像
        img_bytes = self.h5_file['images'][actual_idx]

        # h5 中的图像是 RGB 数组 [3, H, W]
        if isinstance(img_bytes, np.ndarray):
            # 直接是 numpy 数组
            img = img_bytes.transpose(1, 2, 0)  # [3, H, W] -> [H, W, 3]
        else:
            # 需要解码
            img = Image.open(io.BytesIO(img_bytes))

        # 转换为 PIL Image（如果是 numpy）
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img.astype('uint8'))

        # 应用变换
        if self.transform:
            img = self.transform(img)

        # 返回 local idx（相对于子集的索引）用于 F_buffer 索引
        return img, idx

    def __del__(self):
        if self._h5_file is not None:
            self._h5_file.close()


class DCMHTextDataset(Dataset):
    """
    DCMH 文本标签数据集。

    存储 multi-hot 标签向量。
    """

    def __init__(self, h5_path, indices=None):
        """
        初始化数据集。

        参数：
            h5_path: FLICKR-25K.mat 文件路径
            indices: 要使用的样本索引
        """
        self.h5_path = h5_path
        self.indices = indices
        self._h5_file = None

    @property
    def h5_file(self):
        if self._h5_file is None:
            self._h5_file = h5py.File(self.h5_path, 'r')
        return self._h5_file

    def __len__(self):
        if self.indices is not None:
            return len(self.indices)
        return len(self.h5_file['YAll'])

    def __getitem__(self, idx):
        if self.indices is not None:
            actual_idx = self.indices[idx]
        else:
            actual_idx = idx

        # 读取标签（YAll 是 multi-hot 向量）
        tags = self.h5_file['YAll'][actual_idx]
        labels = self.h5_file['LAll'][actual_idx]

        # 转换为 tensor 并添加维度 [1, y_dim, 1]
        tags_tensor = torch.from_numpy(tags).float().unsqueeze(0).unsqueeze(-1)
        labels_tensor = torch.from_numpy(labels).float()

        # 返回 local idx（相对于子集的索引）
        return tags_tensor, labels_tensor, idx

    def __del__(self):
        if self._h5_file is not None:
            self._h5_file.close()


def create_dataloaders(h5_path, batch_size=128, num_workers=0):
    """
    创建按需加载的 DataLoader。

    参数：
        h5_path: FLICKR-25K.mat 文件路径
        batch_size: 批次大小
        num_workers: 数据加载线程数

    返回:
        train_loader, query_loader, retrieval_loader
    """
    from torchvision import transforms

    # 图像变换（与 reference 一致：只用 ToTensor，不做额外归一化）
    # DCMH 模型内部 forward 会做 x - self.mean，所以直接传入 [0, 1] 范围的 tensor
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),  # 转为 [0, 1] tensor，模型内部会减 mean
    ])

    # 数据划分索引
    query_size = 2000
    training_size = 10000
    database_size = 18015

    query_indices = np.arange(0, query_size)
    train_indices = np.arange(query_size, query_size + training_size)
    retrieval_indices = np.arange(query_size, query_size + database_size)

    # 创建数据集
    train_img_dataset = DCMHImageDataset(h5_path, train_indices, transform)
    query_img_dataset = DCMHImageDataset(h5_path, query_indices, transform)
    retrieval_img_dataset = DCMHImageDataset(h5_path, retrieval_indices, transform)

    train_txt_dataset = DCMHTextDataset(h5_path, train_indices)
    query_txt_dataset = DCMHTextDataset(h5_path, query_indices)
    retrieval_txt_dataset = DCMHTextDataset(h5_path, retrieval_indices)

    # 创建 DataLoader
    train_img_loader = DataLoader(train_img_dataset, batch_size=batch_size,
                                   shuffle=True, num_workers=num_workers)
    query_img_loader = DataLoader(query_img_dataset, batch_size=batch_size,
                                   shuffle=False, num_workers=num_workers)
    retrieval_img_loader = DataLoader(retrieval_img_dataset, batch_size=batch_size,
                                       shuffle=False, num_workers=num_workers)

    return (train_img_loader, train_txt_dataset), \
           (query_img_loader, query_txt_dataset), \
           (retrieval_img_loader, retrieval_txt_dataset)


if __name__ == '__main__':
    # 测试 Dataset
    print("测试 DCMH 数据集类")

    h5_path = './data/FLICKR-25K.mat'

    # 测试图像数据集
    img_dataset = DCMHImageDataset(h5_path, indices=list(range(10)))
    print(f"图像数据集长度：{len(img_dataset)}")

    img, idx = img_dataset[0]
    print(f"单张图像形状：{img.shape if hasattr(img, 'shape') else 'PIL Image'}")

    # 测试文本数据集
    txt_dataset = DCMHTextDataset(h5_path, indices=list(range(10)))
    tags, labels, idx = txt_dataset[0]
    print(f"标签形状：{tags.shape}, {labels.shape}")

    print("数据集测试完成！")
