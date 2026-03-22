"""
DCMH 基础模块

提供 DCMH 模型的基础类，封装 nn.Module 提供模型保存和加载功能。
"""

import torch
import time


class DCMHBasicModule(torch.nn.Module):
    """
    封装 nn.Module，主要提供 save 和 load 两个方法。
    """
    def __init__(self):
        super(DCMHBasicModule, self).__init__()
        self.module_name = str(type(self))

    def load(self, path, use_gpu=False):
        """
        可加载指定路径的模型。

        参数：
            path: 模型文件路径
            use_gpu: 是否使用 GPU 加载
        """
        if not use_gpu:
            self.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage))
        else:
            self.load_state_dict(torch.load(path))

    def save(self, name=None):
        """
        保存模型，默认使用"模型名字 + 时间"作为文件名。

        参数：
            name: 可选的文件名

        返回：
            保存的文件名
        """
        if name is None:
            prefix = self.module_name + '_'
            name = time.strftime(prefix + '%m%d_%H:%M:%S.pth')
        torch.save(self.state_dict(), 'checkpoints/' + name)
        return name

    def forward(self, *input):
        """前向传播，由子类实现。"""
        pass
