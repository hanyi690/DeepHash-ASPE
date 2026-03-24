"""
DCMH 文本模块

使用两个卷积层处理文本标签，用于文本哈希编码。
"""

import torch
from torch import nn
from torch.nn import functional as F
from .dcmh_basic import DCMHBasicModule

LAYER1_NODE = 8192


def weights_init(m):
    """初始化卷积层权重。

    与 MATLAB 一致，使用 0.01 标准差。
    """
    if type(m) == nn.Conv2d:
        nn.init.normal_(m.weight.data, 0.0, 0.01)
        nn.init.normal_(m.bias.data, 0.0, 0.01)


class DCMHTextModule(DCMHBasicModule):
    """
    DCMH 文本编码模块。

    使用两个卷积层处理文本标签：
    1. Conv2d(1, 8192, kernel_size=(y_dim, 1)) - 将标签向量转换为高维表示
    2. Conv2d(8192, bit, kernel_size=1) - 投影到哈希码空间
    """

    def __init__(self, y_dim, bit):
        """
        初始化 DCMH 文本模块。

        参数：
            y_dim: 标签维度（词汇表大小或标签数量）
            bit: 哈希码位数
        """
        super(DCMHTextModule, self).__init__()
        self.module_name = "dcmh_text_model"

        # full-conv layers
        self.conv1 = nn.Conv2d(1, LAYER1_NODE, kernel_size=(y_dim, 1), stride=(1, 1))
        self.conv2 = nn.Conv2d(LAYER1_NODE, bit, kernel_size=1, stride=(1, 1))
        self.apply(weights_init)

    def forward(self, x):
        """
        文本模块的前向传播。

        参数：
            x: 输入张量，期望形状为 [batch, 1, y_dim, 1]

        返回：
            哈希码张量 [batch, bit]
        """
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = x.squeeze()
        return x


def build_dcmh_text_model(y_dim, bit):
    """
    构建 DCMH 文本模型。

    参数：
        y_dim: 标签维度
        bit: 哈希码位数

    返回：
        DCMHTextModule 实例
    """
    return DCMHTextModule(y_dim=y_dim, bit=bit)


if __name__ == "__main__":
    # 测试文本模块
    print("测试 DCMH 文本模块")

    # 创建模型
    y_dim = 1000  # 标签数量
    bit = 64
    model = DCMHTextModule(y_dim=y_dim, bit=bit)

    # 测试前向传播
    batch_size = 4
    dummy_labels = torch.zeros(batch_size, 1, y_dim, 1)
    for i in range(batch_size):
        num_active = torch.randint(1, 10, (1,)).item()
        active_indices = torch.randperm(y_dim)[:num_active]
        dummy_labels[i, 0, active_indices, 0] = 1.0

    print(f"\n输入形状：{dummy_labels.shape}")
    print(f"标签维度：{y_dim}")
    print(f"哈希码位数：{bit}")

    output = model(dummy_labels)
    print(f"输出形状：{output.shape}")
    print(f"输出范围：[{output.min():.4f}, {output.max():.4f}]")

    num_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n总参数数量：{num_params:,}")
    print(f"可训练参数数量：{trainable_params:,}")

    print("\nDCMH 文本模块测试完成！")
