"""
DCMH 图像模块

基于 AlexNet 风格的 CNN 架构，用于图像哈希编码。
"""

import torch
from torch import nn
from .dcmh_basic import DCMHBasicModule


class DCMHImageModule(DCMHBasicModule):
    """
    DCMH 图像编码模块。

    使用 AlexNet 风格的 CNN 架构：
    - 5 个卷积层
    - 2 个全卷积层
    - 1 个线性分类器输出哈希码

    架构细节：
    1. conv1: Conv2d(3, 64, kernel_size=11, stride=4) + ReLU + LocalResponseNorm + MaxPool
    2. conv2: Conv2d(64, 256, kernel_size=5, padding=2) + ReLU + LocalResponseNorm + MaxPool
    3. conv3: Conv2d(256, 256, kernel_size=3, padding=1) + ReLU
    4. conv4: Conv2d(256, 256, kernel_size=3, padding=1) + ReLU
    5. conv5: Conv2d(256, 256, kernel_size=3, padding=1) + ReLU + MaxPool
    6. full_conv6: Conv2d(256, 4096, kernel_size=6) + ReLU
    7. full_conv7: Conv2d(4096, 4096, kernel_size=1) + ReLU
    8. classifier: Linear(4096, bit)
    """

    def __init__(self, bit, pretrain_model=None):
        """
        初始化 DCMH 图像模块。

        参数：
            bit: 哈希码位数
            pretrain_model: 可选的预训练模型路径
        """
        super(DCMHImageModule, self).__init__()
        self.module_name = "dcmh_image_model"

        # CNN 特征提取器
        self.features = nn.Sequential(
            # conv1
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=11, stride=4),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=2, k=2),
            nn.ZeroPad2d((0, 1, 0, 1)),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2),

            # conv2
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=2, k=2),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2),

            # conv3
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            # conv4
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            # conv5
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(0, 0)),

            # full_conv6
            nn.Conv2d(in_channels=256, out_channels=4096, kernel_size=6),
            nn.ReLU(inplace=True),

            # full_conv7
            nn.Conv2d(in_channels=4096, out_channels=4096, kernel_size=1),
            nn.ReLU(inplace=True),
        )

        # 分类器输出哈希码
        self.classifier = nn.Linear(in_features=4096, out_features=bit)
        self.classifier.weight.data = torch.randn(bit, 4096) * 0.01
        self.classifier.bias.data = torch.randn(bit) * 0.01

        # 图像归一化均值
        self.mean = torch.zeros(3, 224, 224)

        # 加载预训练模型
        if pretrain_model:
            self._init(pretrain_model)

    def _init(self, data):
        """
        从预训练数据初始化权重。

        参数：
            data: 预训练模型数据字典
        """
        weights = data['layers'][0]
        self.mean = torch.from_numpy(
            data['normalization'][0][0][0].transpose()
        ).type(torch.float)

        for k, v in self.features.named_children():
            k = int(k)
            if isinstance(v, nn.Conv2d):
                if k > 1:
                    k -= 1
                v.weight.data = torch.from_numpy(
                    weights[k][0][0][0][0][0].transpose()
                )
                v.bias.data = torch.from_numpy(
                    weights[k][0][0][0][0][1].reshape(-1)
                )

    def forward(self, x):
        """
        图像模块的前向传播。

        参数：
            x: 输入图像张量 [batch, 3, 224, 224]

        返回：
            哈希码张量 [batch, bit]
        """
        # 减去均值进行归一化
        if x.is_cuda:
            x = x - self.mean.cuda()
        else:
            x = x - self.mean

        # 特征提取
        x = self.features(x)
        x = x.squeeze()

        # 输出哈希码
        x = self.classifier(x)
        return x


def build_dcmh_image_model(bit, pretrain_model=None):
    """
    构建 DCMH 图像模型。

    参数：
        bit: 哈希码位数
        pretrain_model: 可选的预训练模型路径

    返回：
        DCMHImageModule 实例
    """
    return DCMHImageModule(bit=bit, pretrain_model=pretrain_model)


if __name__ == "__main__":
    # 测试图像模块
    print("测试 DCMH 图像模块")

    # 创建模型
    bit = 64
    model = DCMHImageModule(bit=bit, pretrain_model=None)

    # 测试前向传播
    batch_size = 4
    dummy_input = torch.randn(batch_size, 3, 224, 224)

    print(f"\n输入形状：{dummy_input.shape}")

    # 获取输出
    output = model(dummy_input)
    print(f"输出形状：{output.shape}")
    print(f"输出范围：[{output.min():.4f}, {output.max():.4f}]")

    # 统计参数数量
    num_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n总参数数量：{num_params:,}")
    print(f"可训练参数数量：{trainable_params:,}")

    print("\nDCMH 图像模块测试完成！")
