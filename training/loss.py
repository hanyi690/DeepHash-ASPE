"""
跨模态深度哈希的损失函数

实现用于训练双流模型的各种损失函数：
- 用于模态内学习的对比损失
- 用于相对相似性的三元组损失
- 用于图像-文本对齐的跨模态损失
- 哈希量化损失
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class ContrastiveLoss(nn.Module):
    """
    用于学习相似性结构的对比损失。

    将相似项拉近，将不相似项推远。
    """

    def __init__(self, margin: float = 1.0):
        """
        初始化对比损失。

        参数：
            margin: 不相似对的边界值
        """
        super().__init__()
        self.margin = margin

    def forward(self,
                features1: torch.Tensor,
                features2: torch.Tensor,
                labels: torch.Tensor) -> torch.Tensor:
        """
        计算对比损失。

        参数：
            features1: [N, d] 特征向量
            features2: [N, d] 特征向量
            labels: [N] 标签（1表示相似，0表示不相似）

        返回：
            标量损失
        """
        # 计算L2距离
        distances = F.pairwise_distance(features1, features2)

        # 对比损失
        loss_similar = labels * distances ** 2
        loss_dissimilar = (1 - labels) * F.relu(self.margin - distances) ** 2

        loss = (loss_similar + loss_dissimilar).mean()

        return loss


class TripletLoss(nn.Module):
    """
    用于学习相对相似性的三元组损失。

    对于每个锚点，强制执行：
    distance(anchor, positive) < distance(anchor, negative) - margin
    """

    def __init__(self, margin: float = 1.0):
        """
        初始化三元组损失。

        参数：
            margin: 三元组损失的边界值
        """
        super().__init__()
        self.margin = margin

    def forward(self,
                anchors: torch.Tensor,
                positives: torch.Tensor,
                negatives: torch.Tensor) -> torch.Tensor:
        """
        计算三元组损失。

        参数：
            anchors: [N, d] 锚点特征
            positives: [N, d] 正样本特征
            negatives: [N, d] 负样本特征

        返回：
            标量损失
        """
        # 计算距离
        pos_dist = F.pairwise_distance(anchors, positives)
        neg_dist = F.pairwise_distance(anchors, negatives)

        # 三元组损失
        loss = F.relu(pos_dist - neg_dist + self.margin)

        return loss.mean()


class CrossModalLoss(nn.Module):
    """
    用于图像-文本对齐的跨模态损失。

    强制对应的图像-文本对相似，非对不相似。
    """

    def __init__(self, margin: float = 0.5, temperature: float = 0.07):
        """
        初始化跨模态损失。

        参数：
            margin: 不相似对的边界值
            temperature: softmax的温度参数（InfoNCE损失）
        """
        super().__init__()
        self.margin = margin
        self.temperature = temperature

    def contrastive_loss(self,
                        image_features: torch.Tensor,
                        text_features: torch.Tensor,
                        labels: torch.Tensor) -> torch.Tensor:
        """
        图像和文本特征之间的对比损失。

        参数：
            image_features: [N, d] 图像特征
            text_features: [N, d] 文本特征
            labels: [N] 匹配标签（如果匹配则为1，否则为0）

        返回：
            标量损失
        """
        # 计算余弦相似度
        image_features = F.normalize(image_features, dim=1)
        text_features = F.normalize(text_features, dim=1)

        similarities = torch.sum(image_features * text_features, dim=1)

        # 对比损失
        loss_similar = labels * (1 - similarities)
        loss_dissimilar = (1 - labels) * F.relu(similarities - self.margin)

        loss = (loss_similar + loss_dissimilar).mean()

        return loss

    def infonce_loss(self,
                    image_features: torch.Tensor,
                    text_features: torch.Tensor) -> torch.Tensor:
        """
        用于跨模态对齐的InfoNCE（对比）损失。

        参数：
            image_features: [N, d] 图像特征
            text_features: [N, d] 文本特征（按行配对）

        返回：
            标量损失
        """
        # 归一化特征
        image_features = F.normalize(image_features, dim=1)
        text_features = F.normalize(text_features, dim=1)

        # 计算相似度矩阵 [N, N]
        similarity_matrix = torch.mm(image_features, text_features.t()) / self.temperature

        # 正样本对在对角线上
        batch_size = image_features.shape[0]
        labels = torch.arange(batch_size, device=image_features.device)

        # 交叉熵损失
        loss = F.cross_entropy(similarity_matrix, labels)

        return loss

    def forward(self,
                image_features: torch.Tensor,
                text_features: torch.Tensor,
                labels: Optional[torch.Tensor] = None,
                loss_type: str = 'infonce') -> torch.Tensor:
        """
        计算跨模态损失。

        参数：
            image_features: [N, d] 图像特征
            text_features: [N, d] 文本特征
            labels: 可选的匹配标签（用于对比损失）
            loss_type: 'infonce'或'contrastive'

        返回：
            标量损失
        """
        if loss_type == 'infonce':
            return self.infonce_loss(image_features, text_features)
        elif loss_type == 'contrastive':
            if labels is None:
                # 假设按行配对
                labels = torch.ones(image_features.shape[0], device=image_features.device)
            return self.contrastive_loss(image_features, text_features, labels)
        else:
            raise ValueError(f"未知的损失类型：{loss_type}")


class HashQuantizationLoss(nn.Module):
    """
    二进制哈希码的量化损失。

    鼓励连续特征接近二进制值（-1或+1）。
    """

    def __init__(self):
        """初始化量化损失。"""
        super().__init__()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        计算量化损失。

        参数：
            features: [N, d] 连续特征

        返回：
            标量损失（到二进制值的均方距离）
        """
        # 计算特征的符号
        binary_codes = torch.sign(features)

        # 特征和二进制码之间的均方误差
        loss = F.mse_loss(features, binary_codes.detach())

        return loss


class CombinedHashLoss(nn.Module):
    """
    用于训练深度哈希模型的组合损失。

    组合了：
    1. 跨模态对齐损失（图像-文本相似性）
    2. 模态内对比损失
    3. 哈希量化损失
    """

    def __init__(self,
                 margin: float = 0.5,
                 temperature: float = 0.07,
                 alpha: float = 1.0,
                 beta: float = 0.1,
                 gamma: float = 0.01):
        """
        初始化组合损失。

        参数：
            margin: 对比损失的边界值
            temperature: InfoNCE损失的温度参数
            alpha: 跨模态损失的权重
            beta: 模态内损失的权重
            gamma: 量化损失的权重
        """
        super().__init__()
        self.cross_modal_loss = CrossModalLoss(margin, temperature)
        self.contrastive_loss = ContrastiveLoss(margin)
        self.quantization_loss = HashQuantizationLoss()

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self,
                image_features: torch.Tensor,
                text_features: torch.Tensor,
                labels: Optional[torch.Tensor] = None,
                use_quantization: bool = True) -> Tuple[torch.Tensor, dict]:
        """
        计算组合损失。

        参数：
            image_features: [N, d] 图像特征
            text_features: [N, d] 文本特征
            labels: 可选的匹配标签
            use_quantization: 是否使用量化损失

        返回：
            (total_loss, loss_dict)元组
        """
        losses = {}

        # 跨模态损失（图像-文本对齐）
        cm_loss = self.cross_modal_loss.infonce_loss(image_features, text_features)
        losses['cross_modal'] = cm_loss

        # 总损失
        total_loss = self.alpha * cm_loss

        # 量化损失（可选）
        if use_quantization:
            q_loss_img = self.quantization_loss(image_features)
            q_loss_txt = self.quantization_loss(text_features)
            q_loss = (q_loss_img + q_loss_txt) / 2
            losses['quantization'] = q_loss
            total_loss += self.gamma * q_loss

        return total_loss, losses


class TripletHashLoss(nn.Module):
    """
    基于三元组的哈希损失。

    使用三元组（锚点、正样本、负样本）学习相似性结构。
    """

    def __init__(self, margin: float = 1.0, quantization_weight: float = 0.1):
        """
        初始化三元组哈希损失。

        参数：
            margin: 三元组边界值
            quantization_weight: 量化损失的权重
        """
        super().__init__()
        self.triplet_loss = TripletLoss(margin)
        self.quantization_loss = HashQuantizationLoss()
        self.quantization_weight = quantization_weight

    def forward(self,
                anchors: torch.Tensor,
                positives: torch.Tensor,
                negatives: torch.Tensor,
                use_quantization: bool = True) -> Tuple[torch.Tensor, dict]:
        """
        计算三元组哈希损失。

        参数：
            anchors: [N, d] 锚点特征
            positives: [N, d] 正样本特征
            negatives: [N, d] 负样本特征
            use_quantization: 是否使用量化损失

        返回：
            (total_loss, loss_dict)元组
        """
        losses = {}

        # 三元组损失
        triplet = self.triplet_loss(anchors, positives, negatives)
        losses['triplet'] = triplet

        total_loss = triplet

        # 量化损失
        if use_quantization:
            q_loss = (
                self.quantization_loss(anchors) +
                self.quantization_loss(positives) +
                self.quantization_loss(negatives)
            ) / 3
            losses['quantization'] = q_loss
            total_loss += self.quantization_weight * q_loss

        return total_loss, losses


if __name__ == "__main__":
    # 测试损失函数
    print("测试损失函数")

    batch_size = 8
    feature_dim = 128

    # 生成虚拟特征
    image_features = torch.randn(batch_size, feature_dim)
    text_features = torch.randn(batch_size, feature_dim)
    anchors = torch.randn(batch_size, feature_dim)
    positives = torch.randn(batch_size, feature_dim)
    negatives = torch.randn(batch_size, feature_dim)

    # 测试对比损失
    print("\n=== 对比损失 ===")
    contrastive_loss = ContrastiveLoss(margin=1.0)
    labels = torch.randint(0, 2, (batch_size,))
    loss = contrastive_loss(image_features, text_features, labels)
    print(f"损失：{loss.item():.4f}")

    # 测试三元组损失
    print("\n=== 三元组损失 ===")
    triplet_loss = TripletLoss(margin=1.0)
    loss = triplet_loss(anchors, positives, negatives)
    print(f"损失：{loss.item():.4f}")

    # 测试跨模态损失
    print("\n=== 跨模态损失 ===")
    cm_loss = CrossModalLoss(margin=0.5, temperature=0.07)

    # InfoNCE损失
    loss = cm_loss.infonce_loss(image_features, text_features)
    print(f"InfoNCE损失：{loss.item():.4f}")

    # 对比损失
    labels = torch.ones(batch_size)
    loss = cm_loss.contrastive_loss(image_features, text_features, labels)
    print(f"对比损失：{loss.item():.4f}")

    # 测试量化损失
    print("\n=== 量化损失 ===")
    q_loss = HashQuantizationLoss()
    loss = q_loss(image_features)
    print(f"损失：{loss.item():.4f}")

    # 测试组合损失
    print("\n=== 组合哈希损失 ===")
    combined_loss = CombinedHashLoss(alpha=1.0, beta=0.1, gamma=0.01)
    total_loss, loss_dict = combined_loss(image_features, text_features)
    print(f"总损失：{total_loss.item():.4f}")
    for name, value in loss_dict.items():
        print(f"  {name}: {value.item():.4f}")

    # 测试三元组哈希损失
    print("\n=== 三元组哈希损失 ===")
    triplet_hash_loss = TripletHashLoss(margin=1.0, quantization_weight=0.1)
    total_loss, loss_dict = triplet_hash_loss(anchors, positives, negatives)
    print(f"总损失：{total_loss.item():.4f}")
    for name, value in loss_dict.items():
        print(f"  {name}: {value.item():.4f}")

    print("\n损失函数测试完成！")
