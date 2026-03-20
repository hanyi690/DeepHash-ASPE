"""
文本哈希模型

基于NLP的文本编码器，用于在公共嵌入空间中提取特征向量，
用于跨模态检索。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class TextHashModel(nn.Module):
    """
    基于NLP的文本编码器，用于公共嵌入空间。

    架构：
    1. 词嵌入层
    2. 文本编码器（BiLSTM或Transformer）
    3. 池化（序列上的均值/最大值）
    4. 特征投影到公共空间
    5. 可选的哈希层用于二进制码
    6. L2归一化
    """

    def __init__(self,
                 vocab_size: int,
                 feature_dim: int = 4096,
                 hash_bits: int = 64,
                 embed_dim: int = 300,
                 encoder_type: str = 'transformer',
                 num_layers: int = 6,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 max_seq_length: int = 50,
                 padding_idx: int = 0):
        """
        初始化文本哈希模型。

        参数：
            vocab_size: 词汇表大小
            feature_dim: 输出特征空间的维度
            hash_bits: 二进制哈希码的位数
            embed_dim: 词嵌入维度
            encoder_type: 编码器类型（'transformer'、'lstm'、'gru'）
            num_layers: 编码器层数
            num_heads: 注意力头数（用于transformer）
            dropout: Dropout率
            max_seq_length: 最大序列长度
            padding_idx: 填充标记索引
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.feature_dim = feature_dim
        self.hash_bits = hash_bits
        self.embed_dim = embed_dim
        self.encoder_type = encoder_type
        self.max_seq_length = max_seq_length

        # 词嵌入
        self.embedding = nn.Embedding(
            vocab_size,
            embed_dim,
            padding_idx=padding_idx
        )

        # Transformer的位置编码
        if encoder_type == 'transformer':
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=embed_dim * 4,
                dropout=dropout,
                batch_first=True
            )
            self.encoder = nn.TransformerEncoder(
                encoder_layer,
                num_layers=num_layers
            )

        elif encoder_type == 'lstm':
            self.encoder = nn.LSTM(
                embed_dim,
                embed_dim,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=True,
                dropout=dropout if num_layers > 1 else 0
            )
            # LSTM输出由于是双向的，所以是2*embed_dim
            embed_dim = embed_dim * 2

        elif encoder_type == 'gru':
            self.encoder = nn.GRU(
                embed_dim,
                embed_dim,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=True,
                dropout=dropout if num_layers > 1 else 0
            )
            # GRU输出由于是双向的，所以是2*embed_dim
            embed_dim = embed_dim * 2

        else:
            raise ValueError(f"未知的编码器类型：{encoder_type}")

        # 更新投影用的embed_dim
        self.encoder_output_dim = embed_dim

        # 特征投影到公共空间
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, feature_dim)
        )

        # 可选的哈希层
        if hash_bits > 0:
            self.hash_layer = nn.Linear(feature_dim, hash_bits)
        else:
            self.hash_layer = None

    def forward(self,
                x: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        文本编码器的前向传播。

        参数：
            x: 输入标记索引 [batch, seq_len]
            mask: 注意力掩码 [batch, seq_len]（填充位置为True）

        返回：
            L2归一化的特征向量 [batch, feature_dim]
        """
        # 词嵌入
        embedded = self.embedding(x)  # [batch, seq_len, embed_dim]

        # 编码文本
        if self.encoder_type == 'transformer':
            # 为transformer创建注意力掩码
            # Transformer期望掩码中True表示忽略
            if mask is not None:
                src_key_padding_mask = mask
            else:
                src_key_padding_mask = (x == 0)  # 假设padding_idx=0

            encoded = self.encoder(embedded, src_key_padding_mask=src_key_padding_mask)

        elif self.encoder_type in ['lstm', 'gru']:
            # LSTM/GRU不使用注意力掩码
            encoded, _ = self.encoder(embedded)

        # 池化：序列上的均值（排除填充）
        if mask is not None:
            # 为池化创建掩码 [batch, seq_len, 1]
            pool_mask = (~mask).unsqueeze(-1).float()
            pooled = (encoded * pool_mask).sum(dim=1) / (pool_mask.sum(dim=1) + 1e-9)
        else:
            # 简单均值池化
            pooled = encoded.mean(dim=1)

        # 投影到公共嵌入空间
        projected = self.fc(pooled)  # [batch, feature_dim]

        # L2归一化
        normalized = F.normalize(projected, dim=1)

        return normalized

    def get_hash_codes(self,
                      x: torch.Tensor,
                      mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        生成二进制哈希码（可选）。

        参数：
            x: 输入标记索引 [batch, seq_len]
            mask: 注意力掩码 [batch, seq_len]

        返回：
            二进制哈希码 [batch, hash_bits]
        """
        if self.hash_layer is None:
            raise ValueError("哈希层未初始化。请设置hash_bits > 0。")

        # 获取连续特征
        features = self.forward(x, mask)

        # 应用哈希层和符号函数
        hash_codes = torch.sign(self.hash_layer(features))

        return hash_codes

    def get_features(self,
                    x: torch.Tensor,
                    mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播的别名。

        参数：
            x: 输入标记索引 [batch, seq_len]
            mask: 注意力掩码 [batch, seq_len]

        返回：
            特征向量 [batch, feature_dim]
        """
        return self.forward(x, mask)


class TextHashModelLight(nn.Module):
    """
    轻量级文本哈希模型。

    使用更简单的架构以加快训练速度。
    """

    def __init__(self,
                 vocab_size: int,
                 feature_dim: int = 1024,
                 hash_bits: int = 32,
                 embed_dim: int = 300,
                 num_layers: int = 2,
                 dropout: float = 0.1):
        """
        初始化轻量级模型。

        参数：
            vocab_size: 词汇表大小
            feature_dim: 输出特征空间的维度
            hash_bits: 二进制哈希码的位数
            embed_dim: 词嵌入维度
            num_layers: LSTM层数
            dropout: Dropout率
        """
        super().__init__()

        self.feature_dim = feature_dim
        self.hash_bits = hash_bits

        # 词嵌入
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # LSTM编码器
        self.encoder = nn.LSTM(
            embed_dim,
            embed_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # 特征投影
        self.fc = nn.Sequential(
            nn.Linear(embed_dim * 2, feature_dim),
            nn.ReLU(),
            nn.BatchNorm1d(feature_dim),
            nn.Linear(feature_dim, feature_dim)
        )

        # 哈希层
        self.hash_layer = nn.Linear(feature_dim, hash_bits)

    def forward(self,
                x: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """前向传播。"""
        # 嵌入
        embedded = self.embedding(x)

        # 编码
        encoded, _ = self.encoder(embedded)

        # 池化
        if mask is not None:
            pool_mask = (~mask).unsqueeze(-1).float()
            pooled = (encoded * pool_mask).sum(dim=1) / (pool_mask.sum(dim=1) + 1e-9)
        else:
            pooled = encoded.mean(dim=1)

        # 投影
        projected = self.fc(pooled)
        return F.normalize(projected, dim=1)

    def get_hash_codes(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """生成二进制哈希码。"""
        features = self.forward(x, mask)
        return torch.sign(self.hash_layer(features))


class PretrainedTextEncoder(nn.Module):
    """
    使用预训练transformer的文本编码器（例如BERT、RoBERTa）。

    这利用预训练模型以获得更好的文本理解能力。
    """

    def __init__(self,
                 model_name: str = 'bert-base-uncased',
                 feature_dim: int = 4096,
                 hash_bits: int = 64,
                 dropout: float = 0.1):
        """
        初始化预训练文本编码器。

        参数：
            model_name: HuggingFace模型名称
            feature_dim: 输出特征空间的维度
            hash_bits: 二进制哈希码的位数
            dropout: Dropout率
        """
        super().__init__()

        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError:
            raise ImportError("需要transformers库。请使用以下命令安装：pip install transformers")

        self.model_name = model_name
        self.feature_dim = feature_dim
        self.hash_bits = hash_bits

        # 加载预训练模型
        self.backbone = AutoModel.from_pretrained(model_name)
        backbone_dim = self.backbone.config.hidden_size

        # 特征投影
        self.fc = nn.Sequential(
            nn.Linear(backbone_dim, feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, feature_dim)
        )

        # 哈希层
        if hash_bits > 0:
            self.hash_layer = nn.Linear(feature_dim, hash_bits)
        else:
            self.hash_layer = None

    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播。

        参数：
            input_ids: 输入标记ID [batch, seq_len]
            attention_mask: 注意力掩码 [batch, seq_len]

        返回：
            L2归一化的特征向量 [batch, feature_dim]
        """
        # 获取预训练模型输出
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)

        # 使用[CLS]标记表示或均值池化
        pooled = outputs.last_hidden_state[:, 0]  # [CLS]标记

        # 投影到公共空间
        projected = self.fc(pooled)

        # L2归一化
        return F.normalize(projected, dim=1)

    def get_hash_codes(self,
                      input_ids: torch.Tensor,
                      attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """生成二进制哈希码。"""
        if self.hash_layer is None:
            raise ValueError("哈希层未初始化。请设置hash_bits > 0。")

        features = self.forward(input_ids, attention_mask)
        return torch.sign(self.hash_layer(features))


def build_text_model(config: dict) -> TextHashModel:
    """
    从配置构建文本哈希模型。

    参数：
        config: 模型配置字典

    返回：
        TextHashModel实例
    """
    return TextHashModel(
        vocab_size=config.get('vocab_size', 10000),
        feature_dim=config.get('feature_dim', 4096),
        hash_bits=config.get('hash_bits', 64),
        embed_dim=config.get('embed_dim', 300),
        encoder_type=config.get('encoder_type', 'transformer'),
        num_layers=config.get('num_layers', 6),
        num_heads=config.get('num_heads', 8),
        dropout=config.get('dropout', 0.1),
        max_seq_length=config.get('max_seq_length', 50)
    )


if __name__ == "__main__":
    # 测试文本哈希模型
    print("测试文本哈希模型")

    # 配置
    vocab_size = 10000
    feature_dim = 4096
    batch_size = 4
    seq_len = 20

    # 创建模型
    model = TextHashModel(
        vocab_size=vocab_size,
        feature_dim=feature_dim,
        hash_bits=64,
        embed_dim=300,
        encoder_type='transformer',
        num_layers=2,  # 测试时使用较小的值
        num_heads=4,
        dropout=0.1
    )

    # 虚拟输入
    dummy_input = torch.randint(0, vocab_size, (batch_size, seq_len))
    dummy_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    dummy_mask[:, seq_len//2:] = True  # 掩码后半部分

    print(f"\n输入形状：{dummy_input.shape}")
    print(f"掩码形状：{dummy_mask.shape}")

    # 前向传播
    features = model(dummy_input, dummy_mask)
    print(f"特征形状：{features.shape}")
    print(f"特征范数（应该约为1.0）：{torch.norm(features[0]):.4f}")

    # 哈希码
    hash_codes = model.get_hash_codes(dummy_input, dummy_mask)
    print(f"哈希码形状：{hash_codes.shape}")
    print(f"哈希码（二进制）：{hash_codes[0, :10]}")

    # 计算参数数量
    num_params = sum(p.numel() for p in model.parameters())
    print(f"\n总参数量：{num_params:,}")

    # 测试轻量级模型
    print("\n=== 测试轻量级模型 ===")
    light_model = TextHashModelLight(vocab_size=10000, feature_dim=1024, hash_bits=32)
    light_features = light_model(dummy_input, dummy_mask)
    print(f"轻量级模型特征形状：{light_features.shape}")

    light_params = sum(p.numel() for p in light_model.parameters())
    print(f"轻量级模型参数量：{light_params:,}")
