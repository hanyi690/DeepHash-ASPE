"""
双流哈希模型

统一模型，将图像和文本映射到公共嵌入空间
用于跨模态检索。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, List, Tuple
from core.hashing.image_hash import ImageHashModel
from core.hashing.text_hash import TextHashModel


class DualStreamHashModel(nn.Module):
    """
    用于图文检索的双流深度哈希模型。

    两种模态都映射到具有相同维度的公共嵌入空间，
    通过内积或余弦相似度实现直接比较。

    架构：
    1. 图像流：CNN → 特征 → 投影 → 公共空间
    2. 文本流：NLP 编码器 → 特征 → 投影 → 公共空间
    3. 可选：共享变换以实现更好的对齐

    支持的检索模式：
    - 文本 → 图像：文本查询检索图像
    - 图像 → 文本：图像查询检索文本
    - 图像 → 图像：图像查询检索图像
    - 文本 → 文本：文本查询检索文本
    """

    def __init__(self,
                 vocab_size: int,
                 feature_dim: int = 4096,
                 hash_bits: int = 64,
                 image_backbone: str = 'resnet50',
                 text_encoder_type: str = 'transformer',
                 pretrained: bool = True,
                 dropout: float = 0.5,
                 text_embed_dim: int = 300,
                 text_num_layers: int = 6,
                 text_num_heads: int = 8):
        """
        初始化双流哈希模型。

        参数：
            vocab_size: 文本编码器的词汇表大小
            feature_dim: 公共嵌入空间维度
            hash_bits: 二进制哈希码位数
            image_backbone: 图像的 CNN 骨干网络
            text_encoder_type: 文本的编码器类型
            pretrained: 使用预训练权重
            dropout: Dropout 率
            text_embed_dim: 词嵌入维度
            text_num_layers: 文本编码器层数
            text_num_heads: 注意力头数
        """
        super().__init__()

        self.feature_dim = feature_dim
        self.hash_bits = hash_bits

        # 图像流
        self.image_encoder = ImageHashModel(
            feature_dim=feature_dim,
            hash_bits=hash_bits,
            backbone=image_backbone,
            pretrained=pretrained,
            dropout=dropout
        )

        # 文本流
        self.text_encoder = TextHashModel(
            vocab_size=vocab_size,
            feature_dim=feature_dim,
            hash_bits=hash_bits,
            embed_dim=text_embed_dim,
            encoder_type=text_encoder_type,
            num_layers=text_num_layers,
            num_heads=text_num_heads,
            dropout=dropout
        )

        # 可选：用于进一步对齐的共享全连接层
        self.shared_fc_image = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, feature_dim)
        )

        self.shared_fc_text = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, feature_dim)
        )

        # 用于统一二进制码的哈希层
        if hash_bits > 0:
            self.hash_layer_image = nn.Linear(feature_dim, hash_bits)
            self.hash_layer_text = nn.Linear(feature_dim, hash_bits)
        else:
            self.hash_layer_image = None
            self.hash_layer_text = None

    def forward(self,
                image: Optional[torch.Tensor] = None,
                text: Optional[torch.Tensor] = None,
                text_mask: Optional[torch.Tensor] = None) -> List[torch.Tensor]:
        """
        双流模型的前向传播。

        返回公共嵌入空间中的特征。
        应提供图像、文本或两者。

        参数：
            image: 图像张量 [batch_img, 3, H, W]
            text: 文本标记索引 [batch_txt, seq_len]
            text_mask: 文本注意力掩码 [batch_txt, seq_len]

        返回：
            特征列表 [image_features, text_features]（如果两者都提供）
            或单个特征张量
        """
        features = []

        if image is not None:
            img_feat = self.image_encoder(image)
            img_feat = self.shared_fc_image(img_feat)
            features.append(F.normalize(img_feat, dim=1))

        if text is not None:
            txt_feat = self.text_encoder(text, text_mask)
            txt_feat = self.shared_fc_text(txt_feat)
            features.append(F.normalize(txt_feat, dim=1))

        return features if len(features) > 1 else features[0]

    def get_features(self,
                    modality: str,
                    data: torch.Tensor,
                    mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        获取特定模态的特征。

        参数：
            modality: 'image' 或 'text'
            data: 输入数据（图像张量或文本标记）
            mask: 文本的注意力掩码

        返回：
            公共嵌入空间中的特征向量
        """
        if modality == 'image':
            features = self.image_encoder(data)
            features = self.shared_fc_image(features)
            return F.normalize(features, dim=1)

        elif modality == 'text':
            features = self.text_encoder(data, mask)
            features = self.shared_fc_text(features)
            return F.normalize(features, dim=1)

        else:
            raise ValueError(f"未知模态：{modality}")

    def get_hash_codes(self,
                      modality: str,
                      data: torch.Tensor,
                      mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        为特定模态生成二进制哈希码。

        参数：
            modality: 'image' 或 'text'
            data: 输入数据
            mask: 文本的注意力掩码

        返回：
            二进制哈希码 [batch, hash_bits]
        """
        features = self.get_features(modality, data, mask)

        if modality == 'image':
            hash_layer = self.hash_layer_image
        else:
            hash_layer = self.hash_layer_text

        if hash_layer is None:
            raise ValueError("哈希层未初始化。设置 hash_bits > 0。")

        hash_codes = torch.sign(hash_layer(features))
        return hash_codes

    def compute_similarity(self,
                          features1: torch.Tensor,
                          features2: torch.Tensor,
                          metric: str = 'cosine') -> torch.Tensor:
        """
        计算两组特征之间的相似度。

        参数：
            features1: [N, feature_dim] 特征向量
            features2: [M, feature_dim] 特征向量
            metric: 相似度度量（'cosine', 'inner'）

        返回：
            [N, M] 相似度矩阵
        """
        if metric == 'cosine':
            # 特征已经 L2 归一化，所以内积 = 余弦相似度
            return torch.mm(features1, features2.t())
        elif metric == 'inner':
            return torch.mm(features1, features2.t())
        else:
            raise ValueError(f"未知度量：{metric}")

    def freeze_image_encoder(self):
        """冻结图像编码器参数。"""
        for param in self.image_encoder.parameters():
            param.requires_grad = False
        for param in self.shared_fc_image.parameters():
            param.requires_grad = False

    def freeze_text_encoder(self):
        """冻结文本编码器参数。"""
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        for param in self.shared_fc_text.parameters():
            param.requires_grad = False

    def unfreeze_all(self):
        """解冻所有参数以进行端到端训练。"""
        for param in self.parameters():
            param.requires_grad = True


class DualStreamModelLight(nn.Module):
    """
    轻量级双流模型，用于更快的训练。
    """

    def __init__(self,
                 vocab_size: int,
                 feature_dim: int = 1024,
                 hash_bits: int = 32,
                 dropout: float = 0.3):
        """
        初始化轻量级双流模型。

        参数：
            vocab_size: 词汇表大小
            feature_dim: 公共嵌入维度
            hash_bits: 哈希码位数
            dropout: Dropout 率
        """
        super().__init__()

        from core.hashing.image_hash import ImageHashModelLight
        from core.hashing.text_hash import TextHashModelLight

        self.feature_dim = feature_dim
        self.hash_bits = hash_bits

        # 轻量级编码器
        self.image_encoder = ImageHashModelLight(
            feature_dim=feature_dim,
            hash_bits=hash_bits,
            pretrained=True
        )

        self.text_encoder = TextHashModelLight(
            vocab_size=vocab_size,
            feature_dim=feature_dim,
            hash_bits=hash_bits,
            dropout=dropout
        )

    def forward(self,
                image: Optional[torch.Tensor] = None,
                text: Optional[torch.Tensor] = None,
                text_mask: Optional[torch.Tensor] = None) -> List[torch.Tensor]:
        """前向传播。"""
        features = []

        if image is not None:
            features.append(self.image_encoder(image))

        if text is not None:
            features.append(self.text_encoder(text, text_mask))

        return features if len(features) > 1 else features[0]

    def get_features(self,
                    modality: str,
                    data: torch.Tensor,
                    mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """获取特定模态的特征。"""
        if modality == 'image':
            return self.image_encoder(data)
        elif modality == 'text':
            return self.text_encoder(data, mask)
        else:
            raise ValueError(f"未知模态：{modality}")


def build_dual_stream_model(config: dict) -> DualStreamHashModel:
    """
    从配置构建双流模型。

    参数：
        config: 配置字典

    返回：
        DualStreamHashModel 实例
    """
    return DualStreamHashModel(
        vocab_size=config.get('vocab_size', 10000),
        feature_dim=config.get('feature_dim', 4096),
        hash_bits=config.get('hash_bits', 64),
        image_backbone=config.get('image', {}).get('backbone', 'resnet50'),
        text_encoder_type=config.get('text', {}).get('encoder_type', 'transformer'),
        pretrained=config.get('image', {}).get('pretrained', True),
        dropout=config.get('dropout', 0.5),
        text_embed_dim=config.get('text', {}).get('embed_dim', 300),
        text_num_layers=config.get('text', {}).get('num_layers', 6),
        text_num_heads=config.get('text', {}).get('num_heads', 8)
    )


if __name__ == "__main__":
    # 测试双流模型
    print("测试双流哈希模型")

    # 配置
    vocab_size = 10000
    feature_dim = 1024  # 测试时使用较小的值
    batch_size = 4

    # 创建模型
    model = DualStreamHashModel(
        vocab_size=vocab_size,
        feature_dim=feature_dim,
        hash_bits=32,
        image_backbone='resnet18',  # 较小的骨干网络
        text_encoder_type='lstm',  # 更简单的编码器
        pretrained=False,
        text_num_layers=2
    )

    # 测试输入
    dummy_image = torch.randn(batch_size, 3, 224, 224)
    dummy_text = torch.randint(0, vocab_size, (batch_size, 20))
    dummy_mask = torch.zeros(batch_size, 20, dtype=torch.bool)
    dummy_mask[:, 10:] = True

    print(f"\n图像形状：{dummy_image.shape}")
    print(f"文本形状：{dummy_text.shape}")

    # 测试仅图像的前向传播
    img_features = model(image=dummy_image)
    print(f"图像特征形状：{img_features.shape}")
    print(f"图像特征范数：{torch.norm(img_features[0]):.4f}")

    # 测试仅文本的前向传播
    txt_features = model(text=dummy_text, text_mask=dummy_mask)
    print(f"文本特征形状：{txt_features.shape}")
    print(f"文本特征范数：{torch.norm(txt_features[0]):.4f}")

    # 测试两种模态
    img_feat, txt_feat = model(image=dummy_image, text=dummy_text, text_mask=dummy_mask)
    print(f"\n两种模态 - 图像特征：{img_feat.shape}")
    print(f"两种模态 - 文本特征：{txt_feat.shape}")

    # 测试跨模态相似度
    similarity = model.compute_similarity(txt_feat, img_feat, metric='cosine')
    print(f"\n跨模态相似度矩阵：{similarity.shape}")
    print(f"样本相似度：{similarity[0, :]}")

    # 测试哈希码
    img_hash = model.get_hash_codes('image', dummy_image)
    txt_hash = model.get_hash_codes('text', dummy_text, dummy_mask)
    print(f"\n图像哈希码形状：{img_hash.shape}")
    print(f"文本哈希码形状：{txt_hash.shape}")
    print(f"图像哈希（前 10 位）：{img_hash[0, :10]}")

    # 统计参数数量
    num_params = sum(p.numel() for p in model.parameters())
    print(f"\n总参数数量：{num_params:,}")

    # 测试 get_features 方法
    print("\n=== 测试 get_features ===")
    img_feat_single = model.get_features('image', dummy_image)
    txt_feat_single = model.get_features('text', dummy_text, dummy_mask)
    print(f"图像特征（通过 get_features）：{img_feat_single.shape}")
    print(f"文本特征（通过 get_features）：{txt_feat_single.shape}")

    # 验证相同维度
    assert img_feat_single.shape == txt_feat_single.shape, "特征维度不匹配！"
    print("✓ 图像和文本特征具有相同维度（公共嵌入空间）")

    # 测试轻量级模型
    print("\n=== 测试轻量级模型 ===")
    light_model = DualStreamModelLight(vocab_size=vocab_size, feature_dim=512, hash_bits=16)
    light_img, light_txt = light_model(image=dummy_image, text=dummy_text, text_mask=dummy_mask)
    print(f"轻量级模型 - 图像特征：{light_img.shape}")
    print(f"轻量级模型 - 文本特征：{light_txt.shape}")

    light_params = sum(p.numel() for p in light_model.parameters())
    print(f"轻量级模型参数数量：{light_params:,}")
