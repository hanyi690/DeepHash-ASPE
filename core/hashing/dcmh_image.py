"""
DCMH 图像模块

基于 VGG-F 模型的 CNN 架构，用于图像哈希编码。
"""

import torch
from torch import nn
from .dcmh_basic import DCMHBasicModule


class DCMHImageModule(DCMHBasicModule):
    """
    DCMH 图像编码模块。

    使用 VGG-F 风格的 CNN 架构（从 Caffe 转换）：
    - 5 个卷积层
    - 2 个全卷积层
    - 1 个线性分类器输出哈希码
    """

    def __init__(self, bit, pretrain_model=None):
        """
        初始化 DCMH 图像模块。

        参数：
            bit: 哈希码位数
            pretrain_model: 可选的预训练模型路径 (.mat 格式)
        """
        super(DCMHImageModule, self).__init__()
        self.module_name = "dcmh_image_model"

        self.features = nn.Sequential(
            # 0 conv1
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=11, stride=4),
            # 1 relu1
            nn.ReLU(inplace=True),
            # 2 norm1
            nn.LocalResponseNorm(size=2, k=2),
            # 3 pool1
            nn.ZeroPad2d((0, 1, 0, 1)),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2),
            # 4 conv2
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=5, stride=1, padding=2),
            # 5 relu2
            nn.ReLU(inplace=True),
            # 6 norm2
            nn.LocalResponseNorm(size=2, k=2),
            # 7 pool2
            nn.MaxPool2d(kernel_size=(3, 3), stride=2),
            # 8 conv3
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            # 9 relu3
            nn.ReLU(inplace=True),
            # 10 conv4
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            # 11 relu4
            nn.ReLU(inplace=True),
            # 12 conv5
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            # 13 relu5
            nn.ReLU(inplace=True),
            # 14 pool5
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(0, 0)),
            # 15 full_conv6
            nn.Conv2d(in_channels=256, out_channels=4096, kernel_size=6),
            # 16 relu6
            nn.ReLU(inplace=True),
            # 17 full_conv7
            nn.Conv2d(in_channels=4096, out_channels=4096, kernel_size=1),
            # 18 relu7
            nn.ReLU(inplace=True),
        )
        # fc8
        self.classifier = nn.Linear(in_features=4096, out_features=bit)
        self.classifier.weight.data = torch.randn(bit, 4096) * 0.01
        self.classifier.bias.data = torch.randn(bit) * 0.01
        self.mean = torch.zeros(3, 224, 224)
        if pretrain_model:
            self._init(pretrain_model)

    def _init(self, data):
        """
        从预训练数据初始化权重。

        参数：
            data: 预训练模型数据字典（从 imagenet-vgg-f.mat 加载）
        """
        weights = data['layers'][0]
        self.mean = torch.from_numpy(data['normalization'][0][0][0].transpose()).type(torch.float)
        for k, v in self.features.named_children():
            k = int(k)
            if isinstance(v, nn.Conv2d):
                if k > 1:
                    k -= 1
                v.weight.data = torch.from_numpy(weights[k][0][0][0][0][0].transpose())
                v.bias.data = torch.from_numpy(weights[k][0][0][0][0][1].reshape(-1))

    def forward(self, x):
        """
        图像模块的前向传播。

        参数：
            x: 输入图像张量 [batch, 3, 224, 224]

        返回：
            哈希码张量 [batch, bit]
        """
        if x.is_cuda:
            x = x - self.mean.cuda()
        else:
            x = x - self.mean
        x = self.features(x)
        x = x.squeeze()
        x = self.classifier(x)
        return x


def build_dcmh_image_model(bit, pretrain_model_path=None):
    """
    构建 DCMH 图像模型。

    参数：
        bit: 哈希码位数
        pretrain_model_path: 可选的预训练模型路径

    返回：
        DCMHImageModule 实例
    """
    import scipy.io as scio

    pretrain_data = None
    if pretrain_model_path:
        pretrain_data = scio.loadmat(pretrain_model_path)

    return DCMHImageModule(bit=bit, pretrain_model=pretrain_data)


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
