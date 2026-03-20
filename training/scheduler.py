"""
学习率调度器

用于训练深度哈希模型的自定义学习率调度器。
"""

import torch
from torch.optim.lr_scheduler import _LRScheduler
from typing import List
import math


class WarmupScheduler(_LRScheduler):
    """
    带预热的调度器。

    在预热epoch内线性增加学习率，然后使用基础调度器。
    """

    def __init__(self,
                 optimizer,
                 warmup_epochs: int,
                 base_scheduler,
                 warmup_start_lr: float = 1e-6):
        """
        初始化预热调度器。

        参数：
            optimizer: 优化器
            warmup_epochs: 预热的epoch数
            base_scheduler: 预热后使用的基础调度器
            warmup_start_lr: 预热的起始学习率
        """
        self.warmup_epochs = warmup_epochs
        self.base_scheduler = base_scheduler
        self.warmup_start_lr = warmup_start_lr
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        super().__init__(optimizer)

    def get_lr(self) -> List[float]:
        """获取学习率。"""
        if self.last_epoch < self.warmup_epochs:
            # 线性预热
            alpha = (self.last_epoch + 1) / self.warmup_epochs
            return [self.warmup_start_lr + alpha * (base_lr - self.warmup_start_lr)
                   for base_lr in self.base_lrs]
        else:
            # 使用基础调度器
            return self.base_scheduler.get_lr()

    def step(self, epoch=None):
        """步进调度器。"""
        if self.last_epoch >= self.warmup_epochs:
            self.base_scheduler.step(epoch)
        super().step(epoch)


class CosineAnnealingWarmupScheduler(_LRScheduler):
    """
    带预热的余弦退火。

    结合线性预热和余弦退火。
    """

    def __init__(self,
                 optimizer,
                 warmup_epochs: int,
                 total_epochs: int,
                 min_lr: float = 0,
                 warmup_start_lr: float = 1e-6):
        """
        初始化带预热的余弦退火。

        参数：
            optimizer: 优化器
            warmup_epochs: 预热的epoch数
            total_epochs: 训练的总epoch数
            min_lr: 最小学习率
            warmup_start_lr: 预热的起始学习率
        """
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.warmup_start_lr = warmup_start_lr
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        super().__init__(optimizer)

    def get_lr(self) -> List[float]:
        """获取学习率。"""
        if self.last_epoch < self.warmup_epochs:
            # 线性预热
            alpha = (self.last_epoch + 1) / self.warmup_epochs
            return [self.warmup_start_lr + alpha * (base_lr - self.warmup_start_lr)
                   for base_lr in self.base_lrs]
        else:
            # 余弦退火
            progress = (self.last_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))

            return [self.min_lr + (base_lr - self.min_lr) * cosine_decay
                   for base_lr in self.base_lrs]


class PolynomialDecayScheduler(_LRScheduler):
    """
    多项式衰减学习率调度器。

    LR = (initial_lr - end_lr) * (1 - epoch/max_epochs)^power + end_lr
    """

    def __init__(self,
                 optimizer,
                 max_epochs: int,
                 min_lr: float = 0,
                 power: float = 1.0,
                 warmup_epochs: int = 0,
                 warmup_start_lr: float = 1e-6):
        """
        初始化多项式衰减调度器。

        参数：
            optimizer: 优化器
            max_epochs: 最大epoch数
            min_lr: 最小学习率
            power: 多项式的幂次
            warmup_epochs: 预热的epoch数
            warmup_start_lr: 预热的起始学习率
        """
        self.max_epochs = max_epochs
        self.min_lr = min_lr
        self.power = power
        self.warmup_epochs = warmup_epochs
        self.warmup_start_lr = warmup_start_lr
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        super().__init__(optimizer)

    def get_lr(self) -> List[float]:
        """获取学习率。"""
        if self.last_epoch < self.warmup_epochs:
            # 线性预热
            alpha = (self.last_epoch + 1) / self.warmup_epochs
            return [self.warmup_start_lr + alpha * (base_lr - self.warmup_start_lr)
                   for base_lr in self.base_lrs]
        else:
            # 多项式衰减
            progress = (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
            progress = min(progress, 1.0)
            polynomial_decay = (1 - progress) ** self.power

            return [self.min_lr + (base_lr - self.min_lr) * polynomial_decay
                   for base_lr in self.base_lrs]


def create_scheduler(optimizer,
                     scheduler_type: str = 'step',
                     **kwargs) -> _LRScheduler:
    """
    创建学习率调度器。

    参数：
        optimizer: 优化器
        scheduler_type: 调度器类型（'step'、'cosine'、'plateau'、'polynomial'）
        **kwargs: 调度器的额外参数

    返回：
        学习率调度器
    """
    if scheduler_type == 'step':
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=kwargs.get('step_size', 30),
            gamma=kwargs.get('gamma', 0.1)
        )

    elif scheduler_type == 'multistep':
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=kwargs.get('milestones', [30, 60, 80]),
            gamma=kwargs.get('gamma', 0.1)
        )

    elif scheduler_type == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=kwargs.get('T_max', 100),
            eta_min=kwargs.get('eta_min', 0)
        )

    elif scheduler_type == 'cosine_warmup':
        return CosineAnnealingWarmupScheduler(
            optimizer,
            warmup_epochs=kwargs.get('warmup_epochs', 5),
            total_epochs=kwargs.get('total_epochs', 100),
            min_lr=kwargs.get('min_lr', 0),
            warmup_start_lr=kwargs.get('warmup_start_lr', 1e-6)
        )

    elif scheduler_type == 'polynomial':
        return PolynomialDecayScheduler(
            optimizer,
            max_epochs=kwargs.get('max_epochs', 100),
            min_lr=kwargs.get('min_lr', 0),
            power=kwargs.get('power', 1.0),
            warmup_epochs=kwargs.get('warmup_epochs', 0)
        )

    elif scheduler_type == 'plateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=kwargs.get('mode', 'min'),
            factor=kwargs.get('factor', 0.1),
            patience=kwargs.get('patience', 10)
        )

    elif scheduler_type == 'exponential':
        return torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=kwargs.get('gamma', 0.95)
        )

    else:
        raise ValueError(f"未知的调度器类型：{scheduler_type}")


if __name__ == "__main__":
    # 测试调度器
    import torch.nn as nn

    print("测试学习率调度器")

    # 创建虚拟模型和优化器
    model = nn.Linear(10, 10)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 测试阶梯调度器
    print("\n=== 阶梯调度器 ===")
    scheduler = create_scheduler(optimizer, 'step', step_size=30, gamma=0.1)
    for epoch in range(100):
        scheduler.step()
        if epoch % 20 == 0:
            print(f"Epoch {epoch}: 学习率 = {optimizer.param_groups[0]['lr']:.6f}")

    # 测试带预热的余弦调度器
    print("\n=== 带预热的余弦调度器 ===")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = create_scheduler(optimizer, 'cosine_warmup',
                                warmup_epochs=5, total_epochs=50)
    for epoch in range(50):
        scheduler.step()
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: 学习率 = {optimizer.param_groups[0]['lr']:.6f}")

    # 测试多项式衰减
    print("\n=== 多项式衰减 ===")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = create_scheduler(optimizer, 'polynomial',
                                max_epochs=50, power=2.0, warmup_epochs=5)
    for epoch in range(50):
        scheduler.step()
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: 学习率 = {optimizer.param_groups[0]['lr']:.6f}")

    print("\n调度器测试完成！")
