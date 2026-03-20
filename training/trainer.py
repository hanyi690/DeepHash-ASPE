"""
深度哈希模型的训练器

双流深度哈希模型的训练循环。
"""

import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional, Callable, List, Tuple
import numpy as np
from tqdm import tqdm

from training.loss import CombinedHashLoss
from training.scheduler import create_scheduler
from evaluation.metrics import compute_map


class HashModelTrainer:
    """
    双流深度哈希模型的训练器。

    处理训练循环、验证、检查点保存和早停。
    """

    def __init__(self,
                 model: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 loss_fn: nn.Module,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 scheduler: Optional[object] = None,
                 checkpoint_dir: str = './checkpoints',
                 log_interval: int = 10):
        """
        初始化训练器。

        参数：
            model: 双流哈希模型
            optimizer: 优化器
            loss_fn: 损失函数
            device: 训练设备
            scheduler: 可选的学习率调度器
            checkpoint_dir: 保存检查点的目录
            log_interval: 日志记录间隔
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.scheduler = scheduler
        self.checkpoint_dir = checkpoint_dir
        self.log_interval = log_interval

        # 创建检查点目录
        os.makedirs(checkpoint_dir, exist_ok=True)

        # 训练状态
        self.current_epoch = 0
        self.best_metric = 0.0
        self.train_losses = []
        self.val_metrics = []

    def train_epoch(self,
                   train_loader: DataLoader,
                   epoch: int) -> Dict[str, float]:
        """
        训练一个epoch。

        参数：
            train_loader: 训练数据加载器
            epoch: 当前epoch编号

        返回：
            训练指标字典
        """
        self.model.train()

        total_loss = 0.0
        loss_components = {}
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

        for batch_idx, batch in enumerate(pbar):
            # 将批次移动到设备
            images = batch['image'].to(self.device)
            captions = batch['caption']
            caption_masks = (captions == 0).to(self.device)
            captions = captions.to(self.device)

            # 前向传播
            image_features, text_features = self.model(
                image=images,
                text=captions,
                text_mask=caption_masks
            )

            # 计算损失
            loss, loss_dict = self.loss_fn(
                image_features[0],
                text_features[0],
                use_quantization=True
            )

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # 更新指标
            total_loss += loss.item()
            for key, value in loss_dict.items():
                if key not in loss_components:
                    loss_components[key] = 0.0
                loss_components[key] += value.item()
            num_batches += 1

            # 更新进度条
            pbar.set_postfix({'loss': loss.item()})

        # 计算平均指标
        avg_loss = total_loss / num_batches
        for key in loss_components:
            loss_components[key] /= num_batches

        metrics = {'loss': avg_loss, **loss_components}

        return metrics

    def validate(self,
                val_loader: DataLoader,
                epoch: int) -> Dict[str, float]:
        """
        验证模型。

        参数：
            val_loader: 验证数据加载器
            epoch: 当前epoch编号

        返回：
            验证指标字典
        """
        self.model.eval()

        # 提取特征
        image_features_list = []
        text_features_list = []
        labels_list = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="验证中"):
                images = batch['image'].to(self.device)
                captions = batch['captions']  # 标题列表
                image_ids = batch['image_id']

                # 提取图像特征
                img_feat = self.model.get_features('image', images)
                image_features_list.append(img_feat.cpu().numpy())
                labels_list.extend(image_ids)

                # 对于文本特征，我们需要分词后的标题
                # 暂时跳过验证中的文本特征提取
                # （需要分词）

        # 如果有两种模态则计算指标
        # 这是一个简化版本 - 完整实现会
        # 计算mAP、precision@k等

        metrics = {
            'val_loss': 0.0,  # 占位符
            'mAP': 0.0  # 占位符
        }

        return metrics

    def train(self,
             train_loader: DataLoader,
             val_loader: Optional[DataLoader] = None,
             num_epochs: int = 100,
             early_stopping_patience: Optional[int] = None,
             save_frequency: int = 5) -> Dict:
        """
        完整的训练循环。

        参数：
            train_loader: 训练数据加载器
            val_loader: 可选的验证数据加载器
            num_epochs: 训练的epoch数
            early_stopping_patience: 早停的耐心值
            save_frequency: 每N个epoch保存检查点

        返回：
            训练历史字典
        """
        history = {
            'train_losses': [],
            'val_metrics': [],
            'learning_rates': []
        }

        best_metric = 0.0
        patience_counter = 0

        for epoch in range(1, num_epochs + 1):
            self.current_epoch = epoch

            # 训练
            train_metrics = self.train_epoch(train_loader, epoch)
            self.train_losses.append(train_metrics['loss'])
            history['train_losses'].append(train_metrics['loss'])

            # 日志
            print(f"\nEpoch {epoch}/{num_epochs}")
            print(f"训练损失：{train_metrics['loss']:.4f}")
            for key, value in train_metrics.items():
                if key != 'loss':
                    print(f"  {key}: {value:.4f}")

            # 验证
            if val_loader is not None:
                val_metrics = self.validate(val_loader, epoch)
                self.val_metrics.append(val_metrics)
                history['val_metrics'].append(val_metrics)

                print(f"验证 mAP：{val_metrics['mAP']:.4f}")

                # 检查是否有改进
                current_metric = val_metrics.get('mAP', 0.0)
                if current_metric > best_metric:
                    best_metric = current_metric
                    self.save_checkpoint('best_model.pth')
                    print(f"✓ 新的最佳模型，mAP：{best_metric:.4f}")
                    patience_counter = 0
                else:
                    patience_counter += 1

            # 学习率
            current_lr = self.optimizer.param_groups[0]['lr']
            history['learning_rates'].append(current_lr)

            # 步进调度器
            if self.scheduler is not None:
                if val_loader is not None and hasattr(self.scheduler, 'step'):
                    if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(val_metrics.get('mAP', 0.0))
                    else:
                        self.scheduler.step()
                print(f"学习率：{current_lr:.6f}")

            # 保存检查点
            if epoch % save_frequency == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch}.pth')

            # 早停
            if early_stopping_patience is not None and patience_counter >= early_stopping_patience:
                print(f"\n在{epoch}个epoch后触发早停")
                break

        print(f"\n训练完成！最佳 mAP：{best_metric:.4f}")

        return history

    def save_checkpoint(self, filename: str):
        """保存模型检查点。"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_metrics': self.val_metrics,
            'best_metric': self.best_metric
        }

        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        filepath = os.path.join(self.checkpoint_dir, filename)
        torch.save(checkpoint, filepath)

    def load_checkpoint(self, filename: str):
        """加载模型检查点。"""
        filepath = os.path.join(self.checkpoint_dir, filename)

        checkpoint = torch.load(filepath, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.train_losses = checkpoint['train_losses']
        self.val_metrics = checkpoint['val_metrics']
        self.best_metric = checkpoint['best_metric']

        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        print(f"已从epoch {self.current_epoch}加载检查点")


def create_trainer(model: nn.Module,
                  config: Dict,
                  device: str = 'cpu') -> HashModelTrainer:
    """
    从配置创建训练器。

    参数：
        model: 双流哈希模型
        config: 训练配置
        device: 训练设备

    返回：
        HashModelTrainer实例
    """
    # 创建优化器
    if config['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config.get('weight_decay', 0)
        )
    elif config['optimizer'] == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config.get('weight_decay', 0)
        )
    elif config['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=config['learning_rate'],
            momentum=config.get('momentum', 0.9),
            weight_decay=config.get('weight_decay', 0)
        )
    else:
        raise ValueError(f"未知的优化器：{config['optimizer']}")

    # 创建损失函数
    loss_fn = CombinedHashLoss(
        margin=config.get('margin', 0.5),
        temperature=config.get('temperature', 0.07),
        alpha=config.get('alpha', 1.0),
        beta=config.get('beta', 0.1),
        gamma=config.get('gamma', 0.01)
    )

    # 创建调度器
    scheduler = None
    if config.get('lr_scheduler'):
        scheduler = create_scheduler(
            optimizer,
            config['lr_scheduler'],
            **config.get('scheduler_kwargs', {})
        )

    # 创建训练器
    trainer = HashModelTrainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        scheduler=scheduler,
        checkpoint_dir=config.get('checkpoint_dir', './checkpoints')
    )

    return trainer


if __name__ == "__main__":
    # 测试训练器设置（不实际训练）
    print("测试训练器设置")

    from core.hashing.dual_stream import DualStreamHashModel

    # 创建虚拟模型
    model = DualStreamHashModel(
        vocab_size=1000,
        feature_dim=256,
        hash_bits=32,
        image_backbone='resnet18',
        text_encoder_type='lstm',
        pretrained=False
    )

    # 创建训练配置
    config = {
        'optimizer': 'adam',
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'margin': 0.5,
        'temperature': 0.07,
        'alpha': 1.0,
        'beta': 0.1,
        'gamma': 0.01,
        'lr_scheduler': 'step',
        'scheduler_kwargs': {'step_size': 30, 'gamma': 0.1},
        'checkpoint_dir': './checkpoints'
    }

    # 创建训练器
    trainer = create_trainer(model, config, device='cpu')

    print(f"模型已创建，包含 {sum(p.numel() for p in model.parameters()):,} 个参数")
    print(f"优化器：{trainer.optimizer.__class__.__name__}")
    print(f"损失函数：{trainer.loss_fn.__class__.__name__}")
    print(f"设备：{trainer.device}")

    print("\n训练器设置测试完成！")
