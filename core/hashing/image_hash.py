"""
图像哈希模型

基于CNN的图像编码器,用于在通用嵌入空间中提取特征向量,
用于跨模态检索。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Optional, Tuple


class ImageHashModel(nn.Module):
    """
    基于CNN的图像编码器,用于通用嵌入空间。

    使用预训练主干网络(ResNet50/VGG16)配合自定义投影层,
    将图像映射到特征空间。

    架构:
    1. 预训练CNN主干网络(ResNet50)
    2. 特征投影到公共空间(全连接层)
    3. 可选的哈希层用于生成二进制码
    4. L2归一化得到单位范数特征
    """

    def __init__(self,
                 feature_dim: int = 4096,
                 hash_bits: int = 64,
                 backbone: str = 'resnet50',
                 pretrained: bool = True,
                 dropout: float = 0.5):
        """
        初始化图像哈希模型。

        参数:
            feature_dim: 输出特征空间的维度
            hash_bits: 二进制哈希码的位数(可选)
            backbone: CNN主干网络架构('resnet50', 'vgg16')
            pretrained: 是否使用预训练权重
            dropout: Dropout比率
        """
        super().__init__()

        self.feature_dim = feature_dim
        self.hash_bits = hash_bits
        self.backbone_name = backbone

        # 加载预训练主干网络
        if backbone == 'resnet50':
            backbone_model = models.resnet50(pretrained=pretrained)
            # 移除最后的分类层
            self.backbone = nn.Sequential(*list(backbone_model.children())[:-1])
            backbone_output_dim = 2048

        elif backbone == 'resnet18':
            backbone_model = models.resnet18(pretrained=pretrained)
            self.backbone = nn.Sequential(*list(backbone_model.children())[:-1])
            backbone_output_dim = 512

        elif backbone == 'vgg16':
            backbone_model = models.vgg16(pretrained=pretrained)
            # 移除分类器,保留特征层
            self.backbone = backbone_model.features
            # 自适应池化得到固定尺寸
            self.backbone = nn.Sequential(
                self.backbone,
                nn.AdaptiveAvgPool2d((7, 7)),
                nn.Flatten()
            )
            backbone_output_dim = 512 * 7 * 7

        elif backbone == 'efficientnet_b0':
            backbone_model = models.efficientnet_b0(pretrained=pretrained)
            self.backbone = nn.Sequential(*list(backbone_model.children())[:-1])
            backbone_output_dim = 1280

        else:
            raise ValueError(f"未知的主干网络: {backbone}")

        # 特征投影到公共空间
        self.fc = nn.Sequential(
            nn.Linear(backbone_output_dim, feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, feature_dim)
        )

        # 可选的哈希层用于生成二进制码
        if hash_bits > 0:
            self.hash_layer = nn.Linear(feature_dim, hash_bits)
        else:
            self.hash_layer = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        图像编码器的前向传播。

        参数:
            x: 输入图像张量 [batch, 3, H, W]

        返回:
            L2归一化的特征向量 [batch, feature_dim]
        """
        # 提取CNN特征
        features = self.backbone(x)  # [batch, backbone_dim]

        # 如果需要则展平
        if features.dim() > 2:
            features = features.flatten(1)

        # 投影到公共嵌入空间
        projected = self.fc(features)  # [batch, feature_dim]

        # L2归一化
        normalized = F.normalize(projected, dim=1)

        return normalized

    def get_hash_codes(self, x: torch.Tensor) -> torch.Tensor:
        """
        生成二进制哈希码(可选)。

        参数:
            x: 输入图像张量 [batch, 3, H, W]

        返回:
            二进制哈希码 [batch, hash_bits]
        """
        if self.hash_layer is None:
            raise ValueError("哈希层未初始化。请设置 hash_bits > 0。")

        # 获取连续特征
        features = self.forward(x)

        # 应用哈希层和符号函数
        hash_codes = torch.sign(self.hash_layer(features))

        return hash_codes

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播的别名方法。

        参数:
            x: 输入图像张量 [batch, 3, H, W]

        返回:
            特征向量 [batch, feature_dim]
        """
        return self.forward(x)

    def freeze_backbone(self):
        """冻结主干网络参数(用于仅微调头部)。"""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """解冻主干网络参数(用于端到端训练)。"""
        for param in self.backbone.parameters():
            param.requires_grad = True

    def fine_tune_last_n_blocks(self, n: int = 2):
        """
        仅微调主干网络的最后n个块。

        参数:
            n: 从末尾开始微调的块数量
        """
        # 获取所有参数
        params = list(self.backbone.parameters())

        # 冻结所有参数
        for param in params:
            param.requires_grad = False

        # 解冻最后n层(或块)
        # 这是一个简化的方法 - 具体实现取决于主干网络结构
        for param in params[-n:]:
            param.requires_grad = True


class ImageHashModelLight(nn.Module):
    """
    轻量级图像哈希模型,用于更快的训练/推理。

    使用较小的主干网络(ResNet18)和缩减的维度。
    """

    def __init__(self,
                 feature_dim: int = 1024,
                 hash_bits: int = 32,
                 pretrained: bool = True):
        """
        初始化轻量级模型。

        参数:
            feature_dim: 输出特征空间的维度
            hash_bits: 二进制哈希码的位数
            pretrained: 是否使用预训练权重
        """
        super().__init__()

        self.feature_dim = feature_dim
        self.hash_bits = hash_bits

        # 使用ResNet18作为主干网络
        backbone_model = models.resnet18(pretrained=pretrained)
        self.backbone = nn.Sequential(*list(backbone_model.children())[:-1])
        backbone_output_dim = 512

        # 特征投影
        self.fc = nn.Sequential(
            nn.Linear(backbone_output_dim, feature_dim),
            nn.ReLU(),
            nn.BatchNorm1d(feature_dim),
            nn.Linear(feature_dim, feature_dim)
        )

        # 哈希层
        self.hash_layer = nn.Linear(feature_dim, hash_bits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。"""
        features = self.backbone(x).flatten(1)
        projected = self.fc(features)
        return F.normalize(projected, dim=1)

    def get_hash_codes(self, x: torch.Tensor) -> torch.Tensor:
        """生成二进制哈希码。"""
        features = self.forward(x)
        return torch.sign(self.hash_layer(features))


def build_image_model(config: dict) -> ImageHashModel:
    """
    从配置构建图像哈希模型。

    参数:
        config: 模型配置字典

    返回:
        ImageHashModel实例
    """
    return ImageHashModel(
        feature_dim=config.get('feature_dim', 4096),
        hash_bits=config.get('hash_bits', 64),
        backbone=config.get('backbone', 'resnet50'),
        pretrained=config.get('pretrained', True),
        dropout=config.get('dropout', 0.5)
    )


if __name__ == "__main__":
    # 测试图像哈希模型
    print("Testing Image Hash Model")

    # 创建模型
    model = ImageHashModel(
        feature_dim=4096,
        hash_bits=64,
        backbone='resnet50',
        pretrained=False
    )

    # 测试前向传播
    batch_size = 4
    dummy_input = torch.randn(batch_size, 3, 224, 224)

    print(f"\nInput shape: {dummy_input.shape}")

    # 获取特征
    features = model(dummy_input)
    print(f"Feature shape: {features.shape}")
    print(f"Feature norm (should be ~1.0): {torch.norm(features[0]):.4f}")

    # 获取哈希码
    hash_codes = model.get_hash_codes(dummy_input)
    print(f"Hash codes shape: {hash_codes.shape}")
    print(f"Hash codes (binary): {hash_codes[0, :10]}")

    # 统计参数数量
    num_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {num_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # 测试轻量级模型
    print("\n=== Testing Lightweight Model ===")
    light_model = ImageHashModelLight(feature_dim=1024, hash_bits=32, pretrained=False)
    light_features = light_model(dummy_input)
    print(f"Light model feature shape: {light_features.shape}")

    light_params = sum(p.numel() for p in light_model.parameters())
    print(f"Light model parameters: {light_params:,}")
