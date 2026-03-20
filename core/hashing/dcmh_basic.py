"""
DCMH 基础模块

提供 DCMH 模型的基础类，封装 nn.Module 提供模型保存和加载功能。
"""

import torch
import time


class DCMHBasicModule(torch.nn.Module):
    """
    封装 nn.Module，提供模型保存和加载功能。

    这是 DCMH 模型架构的基础类，为图像和文本模块提供统一的接口。
    """

    def __init__(self):
        super(DCMHBasicModule, self).__init__()
        self.module_name = str(type(self))

    def load(self, path, use_gpu=False):
        """
        加载指定路径的模型。

        参数：
            path: 模型文件路径
            use_gpu: 是否使用 GPU 加载
        """
        if not use_gpu:
            self.load_state_dict(
                torch.load(path, map_location=lambda storage, loc: storage)
            )
        else:
            self.load_state_dict(torch.load(path))

    def save(self, name=None, checkpoint_dir='checkpoints/'):
        """
        保存模型，默认使用"模型名字 + 时间"作为文件名。

        参数：
            name: 可选的文件名
            checkpoint_dir: 保存检查点的目录

        返回：
            保存的文件名
        """
        import os
        os.makedirs(checkpoint_dir, exist_ok=True)

        if name is None:
            prefix = self.module_name + '_'
            name = time.strftime(prefix + '%m%d_%H:%M:%S.pth')

        filepath = os.path.join(checkpoint_dir, name)
        torch.save(self.state_dict(), filepath)
        return name

    def forward(self, *input):
        """前向传播，由子类实现。"""
        pass
