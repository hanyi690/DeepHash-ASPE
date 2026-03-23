"""
DCMH 数据加载工具

加载 FLICKR-25K 等数据集，提供与 reference/DCMH 一致的接口。
"""

import h5py
import scipy.io as scio
from typing import Tuple


def load_data(path: str) -> Tuple:
    """
    加载数据集（兼容 FLICKR-25K.mat 格式）。

    参数:
        path: .mat 文件路径

    返回:
        images, tags, labels
    """
    file = h5py.File(path, 'r')
    # 使用 float32 节省内存（reference 代码在训练时也会转为 float32）
    images = file['images'][:].astype('float32')
    labels = file['LAll'][:]
    tags = file['YAll'][:]
    file.close()
    return images, tags, labels


def load_pretrain_model(path: str) -> dict:
    """
    加载预训练模型（兼容 imagenet-vgg-f.mat 格式）。

    参数:
        path: .mat 文件路径

    返回:
        预训练模型数据字典
    """
    return scio.loadmat(path)


def split_data(images, tags, labels,
               query_size: int = 2000,
               training_size: int = 10000,
               database_size: int = 18015) -> dict:
    """
    划分数据集（与 reference/DCMH/main.py split_data 一致）。

    参数:
        images: 图像数据
        tags: 文本标签
        labels: 类别标签
        query_size: 查询集大小
        training_size: 训练集大小
        database_size: 数据库集大小

    返回:
        包含 query/train/database 划分的字典
    """
    # query: 0-2000
    # train: 2000-12000
    # database (retrieval): 2000-20015
    X = {}
    X['query'] = images[0:query_size]
    X['train'] = images[query_size:query_size + training_size]
    X['retrieval'] = images[query_size:query_size + database_size]

    Y = {}
    Y['query'] = tags[0:query_size]
    Y['train'] = tags[query_size:query_size + training_size]
    Y['retrieval'] = tags[query_size:query_size + database_size]

    L = {}
    L['query'] = labels[0:query_size]
    L['train'] = labels[query_size:query_size + training_size]
    L['retrieval'] = labels[query_size:query_size + database_size]

    return X, Y, L


if __name__ == '__main__':
    # 测试数据加载
    import sys
    if len(sys.argv) > 1:
        data_path = sys.argv[1]
        images, tags, labels = load_data(data_path)
        print(f"images: {images.shape}")
        print(f"tags: {tags.shape}")
        print(f"labels: {labels.shape}")

        # 测试数据划分
        data = split_data(images, tags, labels)
        print(f"\nquery: {data['images']['query'].shape}")
        print(f"train: {data['images']['train'].shape}")
        print(f"database: {data['images']['retrieval'].shape}")
