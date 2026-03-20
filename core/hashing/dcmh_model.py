"""
DCMH 双流哈希模型

整合图像和文本模块，提供统一的双流接口。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
from .dcmh_image import DCMHImageModule
from .dcmh_text import DCMHTextModule


class DCMHModel(nn.Module):
    """
    DCMH (Deep Cross-Modal Hashing) 双流哈希模型。

    该模型将图像和文本映射到共同的哈希空间，支持跨模态检索。

    架构：
    1. 图像流：DCMHImageModule - CNN 特征提取 + 哈希编码
    2. 文本流：DCMHTextModule - 卷积标签处理 + 哈希编码
    3. 共同哈希空间：两种模态输出相同维度的哈希码

    支持的检索模式：
    - 文本 → 图像：使用文本查询检索图像
    - 图像 → 文本：使用图像查询检索文本
    - 图像 → 图像：图像查询检索相似图像
    - 文本 → 文本：文本查询检索相似文本
    """

    def __init__(self,
                 bit: int = 64,
                 y_dim: int = 1000,
                 image_pretrain: Optional[str] = None):
        """
        初始化 DCMH 双流模型。

        参数：
            bit: 哈希码位数（16/32/64 等）
            y_dim: 文本标签维度（标签数量或词汇表大小）
            image_pretrain: 图像模块的预训练模型路径（可选）
        """
        super(DCMHModel, self).__init__()

        self.bit = bit
        self.y_dim = y_dim

        # 图像编码模块
        self.image_module = DCMHImageModule(
            bit=bit,
            pretrain_model=image_pretrain
        )

        # 文本编码模块
        self.text_module = DCMHTextModule(
            y_dim=y_dim,
            bit=bit
        )

    def forward(self,
                image: Optional[torch.Tensor] = None,
                text: Optional[torch.Tensor] = None) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        双流模型的前向传播。

        参数：
            image: 输入图像张量 [batch, 3, 224, 224]（可选）
            text: 输入文本标签张量 [batch, 1, y_dim, 1]（可选）

        返回：
            (image_output, text_output) 元组
            - 如果只提供一种模态，返回该模态的输出
            - 如果提供两种模态，返回两个输出的元组
        """
        outputs = []

        if image is not None:
            img_out = self.image_module(image)
            outputs.append(img_out)

        if text is not None:
            txt_out = self.text_module(text)
            outputs.append(txt_out)

        if len(outputs) == 2:
            return outputs[0], outputs[1]
        elif len(outputs) == 1:
            return outputs[0]
        else:
            raise ValueError("必须提供至少一种模态的输入（image 或 text）")

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """
        编码图像为哈希码。

        参数：
            image: 输入图像张量 [batch, 3, 224, 224]

        返回：
            哈希码张量 [batch, bit]
        """
        return self.image_module(image)

    def encode_text(self, text: torch.Tensor) -> torch.Tensor:
        """
        编码文本标签为哈希码。

        参数：
            text: 输入文本标签张量 [batch, 1, y_dim, 1]

        返回：
            哈希码张量 [batch, bit]
        """
        return self.text_module(text)

    def get_hash_code(self,
                     modality: str,
                     data: torch.Tensor) -> torch.Tensor:
        """
        获取指定模态的哈希码。

        参数：
            modality: 模态类型 ('image' 或 'text')
            data: 输入数据

        返回：
            哈希码张量 [batch, bit]
        """
        if modality == 'image':
            return self.encode_image(data)
        elif modality == 'text':
            return self.encode_text(data)
        else:
            raise ValueError(f"未知的模态：{modality}。必须是 'image' 或 'text'")

    def get_binary_hash(self,
                       modality: str,
                       data: torch.Tensor) -> torch.Tensor:
        """
        获取二进制哈希码（符号函数处理）。

        参数：
            modality: 模态类型 ('image' 或 'text')
            data: 输入数据

        返回：
            二进制哈希码张量 {-1, +1}，形状 [batch, bit]
        """
        continuous_hash = self.get_hash_code(modality, data)
        return torch.sign(continuous_hash)

    def compute_similarity(self,
                          hash1: torch.Tensor,
                          hash2: torch.Tensor) -> torch.Tensor:
        """
        计算两组哈希码之间的相似度。

        使用汉明距离的近似：内积越大，汉明距离越小。

        参数：
            hash1: 哈希码张量 [N, bit]
            hash2: 哈希码张量 [M, bit]

        返回：
            相似度矩阵 [N, M]
        """
        return torch.matmul(hash1, hash2.t())

    def hamming_distance(self,
                        hash1: torch.Tensor,
                        hash2: torch.Tensor) -> torch.Tensor:
        """
        计算汉明距离。

        参数：
            hash1: 二进制哈希码 [N, bit]
            hash2: 二进制哈希码 [M, bit]

        返回：
            汉明距离矩阵 [N, M]
        """
        # 将 {-1, 1} 转换为 {0, 1}
        bin1 = (hash1 + 1) / 2
        bin2 = (hash2 + 1) / 2

        # 计算汉明距离
        # dist = bit - 2 * (bin1 @ bin2.t()) + sum(bin1^2) + sum(bin2^2)
        # 对于二进制码，简化为：
        diff = bin1.unsqueeze(1) - bin2.unsqueeze(0)
        return diff.abs().sum(dim=2)

    def cross_modal_retrieval(self,
                             query_hash: torch.Tensor,
                             database_hash: torch.Tensor,
                             top_k: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        跨模态检索。

        参数：
            query_hash: 查询哈希码 [num_queries, bit]
            database_hash: 数据库哈希码 [num_database, bit]
            top_k: 返回的 Top-K 结果数量

        返回：
            (distances, indices) 元组
            - distances: 距离 [num_queries, top_k]
            - indices: 索引 [num_queries, top_k]
        """
        # 计算相似度
        similarity = self.compute_similarity(query_hash, database_hash)

        # 获取 Top-K
        distances, indices = torch.topk(similarity, k=top_k, dim=1)

        return distances, indices


class DCMHWithQuantization(DCMHModel):
    """
    带量化损失的 DCMH 模型。

    在基础 DCMH 模型上添加了量化感知训练支持。
    """

    def __init__(self,
                 bit: int = 64,
                 y_dim: int = 1000,
                 image_pretrain: Optional[str] = None,
                 quantization_tau: float = 1.0):
        """
        初始化带量化的 DCMH 模型。

        参数：
            bit: 哈希码位数
            y_dim: 文本标签维度
            image_pretrain: 图像预训练路径
            quantization_tau: 量化温度参数
        """
        super(DCMHWithQuantization, self).__init__(
            bit=bit,
            y_dim=y_dim,
            image_pretrain=image_pretrain
        )
        self.tau = quantization_tau

    def quantized_forward(self,
                         image: Optional[torch.Tensor] = None,
                         text: Optional[torch.Tensor] = None) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        量化前向传播，使用 tanh 激活近似符号函数。

        参数：
            image: 输入图像张量（可选）
            text: 输入文本标签张量（可选）

        返回：
            量化后的输出元组
        """
        outputs = []

        if image is not None:
            img_out = self.image_module(image)
            img_out_quantized = torch.tanh(img_out * self.tau)
            outputs.append(img_out_quantized)

        if text is not None:
            txt_out = self.text_module(text)
            txt_out_quantized = torch.tanh(txt_out * self.tau)
            outputs.append(txt_out_quantized)

        if len(outputs) == 2:
            return outputs[0], outputs[1]
        elif len(outputs) == 1:
            return outputs[0]
        else:
            raise ValueError("必须提供至少一种模态的输入")

    def quantization_loss(self,
                         image: Optional[torch.Tensor] = None,
                         text: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        计算量化损失：|| |h| - 1 ||²

        鼓励哈希码接近 {-1, +1}。

        参数：
            image: 输入图像张量（可选）
            text: 输入文本标签张量（可选）

        返回：
            量化损失值
        """
        loss = 0.0
        count = 0

        if image is not None:
            img_out = self.image_module(image)
            loss += F.mse_loss(torch.abs(img_out), torch.ones_like(img_out))
            count += 1

        if text is not None:
            txt_out = self.text_module(text)
            loss += F.mse_loss(torch.abs(txt_out), torch.ones_like(txt_out))
            count += 1

        if count == 0:
            return torch.tensor(0.0, device=self.image_module.mean.device)

        return loss / count


def build_dcmh_model(bit: int = 64,
                    y_dim: int = 1000,
                    image_pretrain: Optional[str] = None,
                    with_quantization: bool = False,
                    quantization_tau: float = 1.0) -> DCMHModel:
    """
    构建 DCMH 模型。

    参数：
        bit: 哈希码位数
        y_dim: 文本标签维度
        image_pretrain: 图像预训练路径
        with_quantization: 是否使用量化训练
        quantization_tau: 量化温度参数

    返回：
        DCMH 模型实例
    """
    if with_quantization:
        return DCMHWithQuantization(
            bit=bit,
            y_dim=y_dim,
            image_pretrain=image_pretrain,
            quantization_tau=quantization_tau
        )
    else:
        return DCMHModel(
            bit=bit,
            y_dim=y_dim,
            image_pretrain=image_pretrain
        )


if __name__ == "__main__":
    # 测试 DCMH 双流模型
    print("测试 DCMH 双流哈希模型")

    # 配置
    bit = 64
    y_dim = 1000
    batch_size = 4

    # 创建模型
    model = DCMHModel(bit=bit, y_dim=y_dim, image_pretrain=None)

    # 测试输入
    dummy_image = torch.randn(batch_size, 3, 224, 224)

    # 模拟 multi-hot 标签
    dummy_text = torch.zeros(batch_size, 1, y_dim, 1)
    for i in range(batch_size):
        num_active = torch.randint(1, 10, (1,)).item()
        active_indices = torch.randperm(y_dim)[:num_active]
        dummy_text[i, 0, active_indices, 0] = 1.0

    print(f"\n图像输入形状：{dummy_image.shape}")
    print(f"文本输入形状：{dummy_text.shape}")
    print(f"哈希码位数：{bit}")

    # 测试图像编码
    img_hash = model.encode_image(dummy_image)
    print(f"\n图像哈希码形状：{img_hash.shape}")
    print(f"图像哈希码范围：[{img_hash.min():.4f}, {img_hash.max():.4f}]")

    # 测试文本编码
    txt_hash = model.encode_text(dummy_text)
    print(f"文本哈希码形状：{txt_hash.shape}")
    print(f"文本哈希码范围：[{txt_hash.min():.4f}, {txt_hash.max():.4f}]")

    # 测试双流前向传播
    img_out, txt_out = model(image=dummy_image, text=dummy_text)
    print(f"\n双流输出 - 图像：{img_out.shape}, 文本：{txt_out.shape}")

    # 测试二进制哈希
    binary_img = model.get_binary_hash('image', dummy_image)
    print(f"\n二进制图像哈希（前 10 位）：{binary_img[0, :10]}")
    print(f"二进制值唯一性：{torch.unique(binary_img)}")

    # 测试相似度计算
    similarity = model.compute_similarity(img_hash, txt_hash)
    print(f"\n跨模态相似度矩阵形状：{similarity.shape}")
    print(f"样本相似度：{similarity[0, :5]}")

    # 测试汉明距离
    hamming_dist = model.hamming_distance(binary_img, binary_img)
    print(f"\n自汉明距离（应为 0）：{hamming_dist[0, 0].item()}")

    # 测试 Top-K 检索
    distances, indices = model.cross_modal_retrieval(img_hash, txt_hash, top_k=3)
    print(f"\n检索结果 - 距离形状：{distances.shape}, 索引形状：{indices.shape}")
    print(f"Top-3 距离：{distances[0]}")

    # 统计参数
    num_params = sum(p.numel() for p in model.parameters())
    print(f"\n总参数数量：{num_params:,}")

    # 测试量化版本
    print("\n=== 测试量化版本 ===")
    quant_model = DCMHWithQuantization(bit=bit, y_dim=y_dim, quantization_tau=1.0)
    quant_loss = quant_model.quantization_loss(image=dummy_image, text=dummy_text)
    print(f"量化损失：{quant_loss.item():.4f}")

    quant_out = quant_model.quantized_forward(image=dummy_image, text=dummy_text)
    print(f"量化输出范围：[{quant_out[0].min():.4f}, {quant_out[0].max():.4f}]")

    print("\nDCMH 双流模型测试完成！")
