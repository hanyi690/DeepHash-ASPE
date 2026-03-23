"""
CNN Image Retrieval (cirtorch)

基于 cnnimageretrieval-pytorch 的图像检索核心模块。

模块结构：
- datasets: 数据集加载和处理
- examples: 训练和测试示例
- layers: 网络层（池化、损失函数等）
- networks: 图像检索网络
- utils: 工具函数
"""

from . import datasets, examples, layers, networks, utils

# 数据集模块
from .datasets import datahelpers, genericdataset, testdataset, traindataset

# 网络层模块
from .layers import functional, loss, normalization, pooling

# 网络模块
from .networks import imageretrievalnet

# 工具模块
from .utils import general, download, evaluate, whiten

__all__ = [
    # 子模块
    'datasets', 'examples', 'layers', 'networks', 'utils',
    # 数据集
    'datahelpers', 'genericdataset', 'testdataset', 'traindataset',
    # 网络层
    'functional', 'loss', 'normalization', 'pooling',
    # 网络
    'imageretrievalnet',
    # 工具
    'general', 'download', 'evaluate', 'whiten',
]