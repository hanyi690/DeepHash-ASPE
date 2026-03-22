"""
Flickr25K 数据集 DCMH 模型训练脚本

使用 Flickr25K 数据集训练 DCMH 模型，支持：
- 完整训练循环
- 验证和 mAP 评估
- 模型保存和检查点
- 训练可视化
"""

import os
import sys
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.flickr25k_dataset import load_flickr25k_data, ArrayDataset
from core.hashing.dcmh_model import DCMHModel
from core.hashing.dcmh_image import DCMHImageModule
from core.hashing.dcmh_text import DCMHTextModule
from training.dcmh_loss import DCMHCombinedLoss, DCMHQuantizationLoss
from reference.DCMH.utils import calc_map_k
from evaluation.metrics import compute_hash_quality


class DCMHTrainer:
    """DCMH 模型训练器。"""

    def __init__(self,
                 data_path: str,
                 bit_dim: int = 64,
                 batch_size: int = 128,
                 lr: float = 1e-4,
                 gamma: float = 1.0,
                 eta: float = 1.0,
                 max_epoch: int = 500,
                 use_gpu: bool = True,
                 valid_interval: int = 10,
                 result_dir: str = 'results/flickr-25k'):
        """
        初始化训练器。

        参数:
            data_path: 数据文件路径
            bit_dim: 哈希码维度
            batch_size: 批次大小
            lr: 学习率
            gamma: 量化损失权重
            eta: 平衡损失权重
            max_epoch: 最大训练轮数
            use_gpu: 是否使用 GPU
            valid_interval: 验证间隔
            result_dir: 结果输出目录
        """
        self.data_path = data_path
        self.bit_dim = bit_dim
        self.batch_size = batch_size
        self.base_lr = lr
        self.gamma = gamma
        self.eta = eta
        self.max_epoch = max_epoch
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.valid_interval = valid_interval
        self.result_dir = Path(result_dir)

        # 创建结果目录
        self.result_dir.mkdir(parents=True, exist_ok=True)

        # 设备
        self.device = torch.device('cuda' if self.use_gpu else 'cpu')
        print(f"使用设备：{self.device}")

        # 加载数据
        self._load_data()

        # 初始化模型
        self._init_model()

        # 初始化损失
        self.criterion = DCMHCombinedLoss(
            quantization_weight=gamma,
            use_info_nce=False
        ).to(self.device)

        # 训练历史
        self.history = {
            'loss': [],
            'map_i2t': [],
            'map_t2i': [],
            'lr': []
        }

        # 最佳模型
        self.best_map = 0.0
        self.best_epoch = 0

    def _load_data(self):
        """加载数据。"""
        print("\n" + "=" * 60)
        print("加载 Flickr25K 数据")
        print("=" * 60)

        data = load_flickr25k_data(
            self.data_path,
            query_size=2000,
            training_size=10000,
            database_size=18015
        )

        # 存储数据
        self.train_images = torch.from_numpy(data['train']['images']).float()
        self.train_tags = torch.from_numpy(data['train']['tags']).float()
        self.train_labels = torch.from_numpy(data['train']['labels']).float()

        self.query_images = torch.from_numpy(data['query']['images']).float()
        self.query_tags = torch.from_numpy(data['query']['tags']).float()
        self.query_labels = torch.from_numpy(data['query']['labels']).float()

        self.database_images = torch.from_numpy(data['database']['images']).float()
        self.database_tags = torch.from_numpy(data['database']['tags']).float()
        self.database_labels = torch.from_numpy(data['database']['labels']).float()

        # 移动到设备
        if self.use_gpu:
            self.train_labels = self.train_labels.cuda()
            self.query_labels = self.query_labels.cuda()
            self.database_labels = self.database_labels.cuda()

        self.n_train = len(self.train_images)
        self.n_query = len(self.query_images)
        self.n_database = len(self.database_images)

        # 文本特征维度
        self.y_dim = self.train_tags.shape[1]

        print(f"\n训练集：{self.n_train}")
        print(f"查询集：{self.n_query}")
        print(f"数据库：{self.n_database}")
        print(f"文本维度：{self.y_dim}")
        print(f"类别数：{self.train_labels.shape[1]}")

    def _init_model(self):
        """初始化模型。"""
        print("\n" + "=" * 60)
        print("初始化 DCMH 模型")
        print("=" * 60)

        self.model = DCMHModel(bit=self.bit_dim, y_dim=self.y_dim)
        self.model.to(self.device)

        # 参数量
        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"总参数量：{n_params:,}")

        # 优化器
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=self.base_lr,
            momentum=0.9,
            weight_decay=1e-5
        )

        # 学习率调度器
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.max_epoch,
            eta_min=1e-6
        )

    def _normalize_tensor(self, x: torch.Tensor) -> torch.Tensor:
        """归一化图像张量。"""
        # 图像已经是 0-255 范围，归一化到 0-1
        return x / 255.0

    def train_epoch(self, epoch: int) -> float:
        """训练一个 epoch。"""
        self.model.train()

        # 随机打乱索引
        indices = torch.randperm(self.n_train)

        total_loss = 0.0
        n_batches = self.n_train // self.batch_size

        # F 和 G 缓冲区
        F_buffer = torch.randn(self.n_train, self.bit_dim)
        G_buffer = torch.randn(self.n_train, self.bit_dim)

        if self.use_gpu:
            F_buffer = F_buffer.cuda()
            G_buffer = G_buffer.cuda()

        # 计算相似度矩阵
        Sim = (self.train_labels @ self.train_labels.t() > 0).float()
        if self.use_gpu:
            Sim = Sim.cuda()

        pbar = tqdm(range(n_batches), desc=f'Epoch {epoch+1}/{self.max_epoch}')

        for i in pbar:
            # 获取批次索引
            batch_indices = indices[i * self.batch_size:(i + 1) * self.batch_size]

            # 批次数据
            batch_images = self._normalize_tensor(self.train_images[batch_indices])
            batch_tags = self.train_tags[batch_indices].unsqueeze(1).unsqueeze(-1)
            batch_labels = self.train_labels[batch_indices]

            if self.use_gpu:
                batch_images = batch_images.cuda()
                batch_tags = batch_tags.cuda()

            # 前向传播 - 图像
            self.optimizer.zero_grad()

            # 图像流
            img_hash = self.model.image_module(batch_images)
            F_buffer[batch_indices] = img_hash.detach()

            # 计算图像损失
            F = F_buffer
            G = G_buffer

            # 相似度 (batch_size, n_train)
            S = (batch_labels @ self.train_labels.t() > 0).float()
            if self.use_gpu:
                S = S.cuda()

            # 交叉模态损失
            theta = 0.5 * img_hash @ G.t()
            logloss = -torch.sum(S * theta - torch.log(1 + torch.exp(theta) + 1e-10))

            # 量化损失
            B = torch.sign(F + G)
            quant_loss = torch.sum(torch.pow(B[batch_indices] - img_hash, 2))

            # 平衡损失
            ones = torch.ones(self.batch_size, 1)
            ones_ = torch.ones(self.n_train - self.batch_size, 1)
            unupdated_indices = torch.tensor([j for j in range(self.n_train) if j not in batch_indices.cpu().tolist()])
            if self.use_gpu:
                ones = ones.cuda()
                ones_ = ones_.cuda()
                unupdated_indices = unupdated_indices.cuda()

            balance_loss = torch.sum(torch.pow(
                img_hash.t() @ ones + F[unupdated_indices].t() @ ones_, 2
            ))

            loss = logloss + self.gamma * quant_loss + self.eta * balance_loss
            loss = loss / (self.batch_size * self.n_train)

            loss.backward()
            self.optimizer.step()

            # 文本流
            self.optimizer.zero_grad()

            batch_tags_input = batch_tags
            txt_hash = self.model.text_module(batch_tags_input)
            G_buffer[batch_indices] = txt_hash.detach()

            # 更新 F 和 G
            F = F_buffer
            G = G_buffer

            # 文本损失
            theta_txt = 0.5 * txt_hash @ F.t()
            logloss_txt = -torch.sum(S * theta_txt - torch.log(1 + torch.exp(theta_txt) + 1e-10))

            B = torch.sign(F + G)
            quant_loss_txt = torch.sum(torch.pow(B[batch_indices] - txt_hash, 2))

            balance_loss_txt = torch.sum(torch.pow(
                txt_hash.t() @ ones + G[unupdated_indices].t() @ ones_, 2
            ))

            loss_txt = logloss_txt + self.gamma * quant_loss_txt + self.eta * balance_loss_txt
            loss_txt = loss_txt / (self.batch_size * self.n_train)

            loss_txt.backward()
            self.optimizer.step()

            # 更新 B
            B = torch.sign(F_buffer + G_buffer)

            total_loss += loss.item() + loss_txt.item()

            pbar.set_postfix({'loss': f'{total_loss / (i + 1):.4f}'})

        return total_loss / n_batches

    @torch.no_grad()
    def validate(self) -> Tuple[float, float]:
        """验证模型。"""
        self.model.eval()

        # 生成查询哈希码
        query_img_input = self._normalize_tensor(self.query_images)
        query_txt_input = self.query_tags.unsqueeze(1).unsqueeze(-1)

        if self.use_gpu:
            query_img_input = query_img_input.cuda()
            query_txt_input = query_txt_input.cuda()

        qBX = torch.sign(self.model.image_module(query_img_input))
        qBY = torch.sign(self.model.text_module(query_txt_input))

        # 生成数据库哈希码
        db_img_input = self._normalize_tensor(self.database_images)
        db_txt_input = self.database_tags.unsqueeze(1).unsqueeze(-1)

        if self.use_gpu:
            db_img_input = db_img_input.cuda()
            db_txt_input = db_txt_input.cuda()

        rBX = torch.sign(self.model.image_module(db_img_input))
        rBY = torch.sign(self.model.text_module(db_txt_input))

        # 计算 mAP
        map_i2t = calc_map_k(qBX, rBY, self.query_labels, self.database_labels)
        map_t2i = calc_map_k(qBY, rBX, self.query_labels, self.database_labels)

        return map_i2t.item(), map_t2i.item()

    @torch.no_grad()
    def generate_hash_codes(self, images: torch.Tensor,
                           tags: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """生成哈希码。"""
        self.model.eval()

        img_input = self._normalize_tensor(images)
        txt_input = tags.unsqueeze(1).unsqueeze(-1)

        if self.use_gpu:
            img_input = img_input.cuda()
            txt_input = txt_input.cuda()

        img_hash = torch.sign(self.model.image_module(img_input))
        txt_hash = torch.sign(self.model.text_module(txt_input))

        return img_hash.cpu().numpy(), txt_hash.cpu().numpy()

    def train(self, resume: bool = False):
        """主训练循环。"""
        print("\n" + "=" * 60)
        print("开始训练 DCMH 模型")
        print("=" * 60)
        print(f"批次大小：{self.batch_size}")
        print(f"学习率：{self.base_lr}")
        print(f"哈希维度：{self.bit_dim}")
        print(f"最大轮数：{self.max_epoch}")
        print(f"验证间隔：{self.valid_interval}")

        start_time = time.time()

        for epoch in range(self.max_epoch):
            # 训练
            train_loss = self.train_epoch(epoch)
            self.history['loss'].append(train_loss)

            # 更新学习率
            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]
            self.history['lr'].append(current_lr)

            # 验证
            if (epoch + 1) % self.valid_interval == 0 or epoch == self.max_epoch - 1:
                map_i2t, map_t2i = self.validate()
                self.history['map_i2t'].append(map_i2t)
                self.history['map_t2i'].append(map_t2i)

                combined_map = (map_i2t + map_t2i) / 2

                print(f"\n验证结果 (Epoch {epoch + 1}):")
                print(f"  MAP(i->t): {map_i2t:.4f}")
                print(f"  MAP(t->i): {map_t2i:.4f}")
                print(f"  平均 MAP: {(map_i2t + map_t2i) / 2:.4f}")

                # 保存最佳模型
                if combined_map > self.best_map:
                    self.best_map = combined_map
                    self.best_epoch = epoch + 1
                    self.save_model('best')
                    print(f"  ✓ 新最佳模型 (MAP: {combined_map:.4f})")

            # 保存检查点
            if (epoch + 1) % 50 == 0:
                self.save_model(f'checkpoint_epoch_{epoch + 1}')

            # 保存最新模型
            self.save_model('latest')

        training_time = time.time() - start_time

        print("\n" + "=" * 60)
        print("训练完成")
        print("=" * 60)
        print(f"训练时间：{training_time / 3600:.2f} 小时")
        print(f"最佳 MAP: {self.best_map:.4f} (Epoch {self.best_epoch})")

        # 保存最终结果
        self.save_results()

        return self.history

    def save_model(self, name: str = 'latest'):
        """保存模型。"""
        checkpoint = {
            'epoch': self.history['loss'].__len__(),
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_map': self.best_map,
            'best_epoch': self.best_epoch,
            'config': {
                'bit_dim': self.bit_dim,
                'batch_size': self.batch_size,
                'lr': self.base_lr,
                'gamma': self.gamma,
                'eta': self.eta,
                'y_dim': self.y_dim
            }
        }

        save_path = self.result_dir / f'dcmh_flickr25k_{name}.pth'
        torch.save(checkpoint, save_path)
        print(f"模型已保存：{save_path}")

    def save_results(self):
        """保存训练结果。"""
        results = {
            'config': {
                'bit_dim': self.bit_dim,
                'batch_size': self.batch_size,
                'lr': self.base_lr,
                'gamma': self.gamma,
                'eta': self.eta,
                'max_epoch': self.max_epoch,
                'y_dim': self.y_dim,
                'n_train': self.n_train,
                'n_query': self.n_query,
                'n_database': self.n_database
            },
            'history': self.history,
            'best_map': self.best_map,
            'best_epoch': self.best_epoch,
            'training_completed': True,
            'completed_at': datetime.now().isoformat()
        }

        results_path = self.result_dir / 'dcmh_training_results.json'
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"训练结果已保存：{results_path}")


def train_flickr25k(data_path: str,
                   result_dir: str = 'results/flickr-25k',
                   **kwargs):
    """
    训练 Flickr25K 数据集的便捷函数。

    参数:
        data_path: 数据文件路径
        result_dir: 结果目录
        **kwargs: 传递给 DCMHTrainer 的参数
    """
    trainer = DCMHTrainer(
        data_path=data_path,
        result_dir=result_dir,
        **kwargs
    )

    history = trainer.train()

    return trainer, history


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='DCMH Flickr25K 训练')
    parser.add_argument('--data', type=str, default='data/flickr25k/FLICKR-25K.mat',
                       help='数据文件路径')
    parser.add_argument('--result-dir', type=str, default='results/flickr-25k',
                       help='结果目录')
    parser.add_argument('--bit', type=int, default=64, help='哈希码维度')
    parser.add_argument('--batch-size', type=int, default=128, help='批次大小')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--epochs', type=int, default=500, help='最大训练轮数')
    parser.add_argument('--gamma', type=float, default=1.0, help='量化损失权重')
    parser.add_argument('--eta', type=float, default=1.0, help='平衡损失权重')
    parser.add_argument('--no-gpu', action='store_true', help='禁用 GPU')

    args = parser.parse_args()

    # 训练
    trainer, history = train_flickr25k(
        data_path=args.data,
        result_dir=args.result_dir,
        bit_dim=args.bit,
        batch_size=args.batch_size,
        lr=args.lr,
        max_epoch=args.epochs,
        gamma=args.gamma,
        eta=args.eta,
        use_gpu=not args.no_gpu
    )

    print("\n训练完成！")
