"""
DCMH 模型的损失函数

实现 DCMH (Deep Cross-Modal Hashing) 模型的特定损失函数：
- 跨模态对比损失
- 量化损失
- 分类损失（可选）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class DCMHQuantizationLoss(nn.Module):
    """
    DCMH 量化损失：|| |h| - 1 ||²

    鼓励哈希码接近二进制值 {-1, +1}。
    """

    def __init__(self):
        super().__init__()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        计算量化损失。

        参数：
            features: [N, d] 连续哈希码

        返回：
            标量损失
        """
        # 计算 |h| - 1 的均方误差
        return F.mse_loss(torch.abs(features), torch.ones_like(features))


class DCMHCrossModalLoss(nn.Module):
    """
    DCMH 跨模态对比损失。

    对于配对的图像 - 文本对，最小化它们之间的距离；
    对于非配对对，最大化距离。
    """

    def __init__(self, margin: float = 1.0):
        """
        初始化跨模态损失。

        参数：
            margin: 对比损失的边界值
        """
        super().__init__()
        self.margin = margin

    def forward(self,
                image_hash: torch.Tensor,
                text_hash: torch.Tensor,
                similarity_matrix: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        计算跨模态对比损失。

        参数：
            image_hash: [N, bit] 图像哈希码
            text_hash: [N, bit] 文本哈希码
            similarity_matrix: [N, N] 可选的相似性矩阵

        返回：
            标量损失
        """
        batch_size = image_hash.shape[0]

        # 如果没有提供相似性矩阵，使用单位矩阵（对角线为配对）
        if similarity_matrix is None:
            similarity_matrix = torch.eye(batch_size, device=image_hash.device)

        # 计算汉明距离的近似（使用内积）
        # 内积越大，汉明距离越小
        similarity = torch.matmul(image_hash, text_hash.t()) / image_hash.shape[1]

        # 对比损失
        # 对于配对样本：最大化相似度
        # 对于非配对样本：最小化相似度（最大化距离）
        pos_loss = -similarity * similarity_matrix
        neg_loss = F.relu(similarity - self.margin) * (1 - similarity_matrix)

        loss = (pos_loss + neg_loss).mean()
        return loss


class DCMHInfoNCELoss(nn.Module):
    """
    DCMH 的 InfoNCE 损失（用于跨模态对齐）。
    """

    def __init__(self, temperature: float = 0.07):
        """
        初始化 InfoNCE 损失。

        参数：
            temperature: 温度参数
        """
        super().__init__()
        self.temperature = temperature
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self,
                image_hash: torch.Tensor,
                text_hash: torch.Tensor) -> torch.Tensor:
        """
        计算 InfoNCE 损失。

        参数：
            image_hash: [N, bit] 图像哈希码
            text_hash: [N, bit] 文本哈希码

        返回：
            标量损失
        """
        # 归一化
        image_hash = F.normalize(image_hash, dim=1)
        text_hash = F.normalize(text_hash, dim=1)

        # 计算相似度矩阵
        similarity = torch.matmul(image_hash, text_hash.t()) / self.temperature

        # 标签是对角线
        batch_size = image_hash.shape[0]
        labels = torch.arange(batch_size, device=image_hash.device)

        # 交叉熵损失
        loss = self.ce_loss(similarity, labels)
        return loss


class DCMHCombinedLoss(nn.Module):
    """
    DCMH 组合损失。

    组合了：
    1. 跨模态对比损失（图像 - 文本对齐）
    2. 量化损失（鼓励二进制输出）
    """

    def __init__(self,
                 margin: float = 1.0,
                 quantization_weight: float = 0.1,
                 use_info_nce: bool = False,
                 temperature: float = 0.07):
        """
        初始化 DCMH 组合损失。

        参数：
            margin: 对比损失的边界值
            quantization_weight: 量化损失的权重
            use_info_nce: 是否使用 InfoNCE 损失
            temperature: InfoNCE 的温度参数
        """
        super().__init__()

        self.cross_modal_loss = DCMHCrossModalLoss(margin=margin)
        self.info_nce_loss = DCMHInfoNCELoss(temperature=temperature)
        self.quantization_loss = DCMHQuantizationLoss()

        self.quantization_weight = quantization_weight
        self.use_info_nce = use_info_nce

    def forward(self,
                image_hash: torch.Tensor,
                text_hash: torch.Tensor,
                similarity_matrix: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, dict]:
        """
        计算 DCMH 组合损失。

        参数：
            image_hash: [N, bit] 图像哈希码
            text_hash: [N, bit] 文本哈希码
            similarity_matrix: [N, N] 可选的相似性矩阵

        返回：
            (total_loss, loss_dict) 元组
        """
        losses = {}

        # 跨模态损失
        if self.use_info_nce:
            cm_loss = self.info_nce_loss(image_hash, text_hash)
        else:
            cm_loss = self.cross_modal_loss(image_hash, text_hash, similarity_matrix)
        losses['cross_modal'] = cm_loss

        # 量化损失
        q_loss_img = self.quantization_loss(image_hash)
        q_loss_txt = self.quantization_loss(text_hash)
        q_loss = (q_loss_img + q_loss_txt) / 2
        losses['quantization'] = q_loss

        # 总损失
        total_loss = cm_loss + self.quantization_weight * q_loss

        return total_loss, losses


class DCMHMarginRankingLoss(nn.Module):
    """
    DCMH 的边界排序损失（Margin Ranking Loss）。

    对于每个锚点，强制执行：
    similarity(anchor, positive) > similarity(anchor, negative) + margin
    """

    def __init__(self, margin: float = 0.5):
        """
        初始化边界排序损失。

        参数：
            margin: 边界值
        """
        super().__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self,
                anchor_hash: torch.Tensor,
                positive_hash: torch.Tensor,
                negative_hash: torch.Tensor) -> torch.Tensor:
        """
        计算边界排序损失。

        参数：
            anchor_hash: [N, bit] 锚点哈希码
            positive_hash: [N, bit] 正样本哈希码
            negative_hash: [N, bit] 负样本哈希码

        返回：
            标量损失
        """
        # 计算相似度
        pos_sim = torch.sum(anchor_hash * positive_hash, dim=1)
        neg_sim = torch.sum(anchor_hash * negative_hash, dim=1)

        # 创建标签（期望 pos_sim > neg_sim）
        target = torch.ones_like(pos_sim)

        loss = self.ranking_loss(pos_sim, neg_sim, target)
        return loss


if __name__ == "__main__":
    # 测试 DCMH 损失函数
    print("测试 DCMH 损失函数")

    batch_size = 8
    bit = 64

    # 生成虚拟哈希码
    image_hash = torch.randn(batch_size, bit)
    text_hash = torch.randn(batch_size, bit)

    # 测试量化损失
    print("\n=== 量化损失 ===")
    q_loss_fn = DCMHQuantizationLoss()
    q_loss = q_loss_fn(image_hash)
    print(f"量化损失：{q_loss.item():.4f}")

    # 测试跨模态损失
    print("\n=== 跨模态对比损失 ===")
    cm_loss_fn = DCMHCrossModalLoss(margin=1.0)
    cm_loss = cm_loss_fn(image_hash, text_hash)
    print(f"跨模态损失：{cm_loss.item():.4f}")

    # 测试 InfoNCE 损失
    print("\n=== InfoNCE 损失 ===")
    info_nce_loss_fn = DCMHInfoNCELoss(temperature=0.07)
    info_nce_loss = info_nce_loss_fn(image_hash, text_hash)
    print(f"InfoNCE 损失：{info_nce_loss.item():.4f}")

    # 测试组合损失
    print("\n=== DCMH 组合损失 ===")
    combined_loss_fn = DCMHCombinedLoss(
        margin=1.0,
        quantization_weight=0.1,
        use_info_nce=False
    )
    total_loss, loss_dict = combined_loss_fn(image_hash, text_hash)
    print(f"总损失：{total_loss.item():.4f}")
    for name, value in loss_dict.items():
        print(f"  {name}: {value.item():.4f}")

    # 测试 InfoNCE 版本
    print("\n=== DCMH 组合损失 (InfoNCE) ===")
    combined_loss_fn_nce = DCMHCombinedLoss(
        quantization_weight=0.1,
        use_info_nce=True,
        temperature=0.07
    )
    total_loss_nce, loss_dict_nce = combined_loss_fn_nce(image_hash, text_hash)
    print(f"总损失：{total_loss_nce.item():.4f}")
    for name, value in loss_dict_nce.items():
        print(f"  {name}: {value.item():.4f}")

    # 测试边界排序损失
    print("\n=== 边界排序损失 ===")
    margin_loss_fn = DCMHMarginRankingLoss(margin=0.5)
    anchor = torch.randn(batch_size, bit)
    positive = torch.randn(batch_size, bit)
    negative = torch.randn(batch_size, bit)
    margin_loss = margin_loss_fn(anchor, positive, negative)
    print(f"边界排序损失：{margin_loss.item():.4f}")

    print("\nDCMH 损失函数测试完成！")
