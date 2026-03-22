"""
Flickr25K 数据集检索评估脚本

评估 DCMH 模型在 Flickr25K 数据集上的检索性能，包括：
- mAP (平均精度均值)
- Precision@K
- Recall@K
- NDCG@K
- 跨模态检索 (图像->文本，文本->图像)
"""

import os
import sys
import json
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from reference.DCMH.utils import calc_map_k
from evaluation.metrics import (
    compute_precision_at_k,
    compute_recall_at_k,
    compute_map,
    compute_hash_quality
)


class Flickr25KEvaluator:
    """Flickr25K 评估器。"""

    def __init__(self,
                 data_path: str,
                 model_path: str,
                 bit_dim: int = 64,
                 batch_size: int = 128,
                 use_gpu: bool = True,
                 result_dir: str = 'results/flickr-25k'):
        """
        初始化评估器。

        参数:
            data_path: 数据文件路径
            model_path: 模型文件路径
            bit_dim: 哈希码维度
            batch_size: 批次大小
            use_gpu: 是否使用 GPU
            result_dir: 结果输出目录
        """
        self.data_path = data_path
        self.model_path = model_path
        self.bit_dim = bit_dim
        self.batch_size = batch_size
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.result_dir = Path(result_dir)

        # 创建结果目录
        self.result_dir.mkdir(parents=True, exist_ok=True)

        # 设备
        self.device = torch.device('cuda' if self.use_gpu else 'cpu')

        # 加载数据和模型
        self._load_data()
        self._load_model()

    def _load_data(self):
        """加载数据。"""
        print("=" * 60)
        print("加载 Flickr25K 数据")
        print("=" * 60)

        import h5py

        with h5py.File(self.data_path, 'r') as f:
            images = f['images'][:].astype('float')
            labels = f['LAll'][:].astype('float')
            tags = f['YAll'][:].astype('float')

        # 转置为 (N, C, H, W)
        if images.ndim == 4:
            images = np.transpose(images, (3, 0, 1, 2))

        # 数据划分
        query_size = 2000
        training_size = 10000
        database_size = 18015

        self.query_images = torch.from_numpy(images[:query_size]).float()
        self.query_tags = torch.from_numpy(tags[:query_size]).float()
        self.query_labels = torch.from_numpy(labels[:query_size]).float()

        self.database_images = torch.from_numpy(
            images[query_size:query_size + database_size]
        ).float()
        self.database_tags = torch.from_numpy(
            tags[query_size:query_size + database_size]
        ).float()
        self.database_labels = torch.from_numpy(
            labels[query_size:query_size + database_size]
        ).float()

        self.y_dim = self.query_tags.shape[1]
        self.n_classes = self.query_labels.shape[1]

        print(f"查询集：{len(self.query_images)}")
        print(f"数据库：{len(self.database_images)}")
        print(f"文本维度：{self.y_dim}")
        print(f"类别数：{self.n_classes}")

    def _load_model(self):
        """加载模型。"""
        print("\n" + "=" * 60)
        print("加载 DCMH 模型")
        print("=" * 60)

        from core.hashing.dcmh_model import DCMHModel

        self.model = DCMHModel(bit=self.bit_dim, y_dim=self.y_dim)

        checkpoint = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)

        print(f"模型已加载：{self.model_path}")
        print(f"最佳 MAP: {checkpoint.get('best_map', 0):.4f}")

    @torch.no_grad()
    def generate_hash_codes(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        生成查询和数据库的哈希码。

        返回:
            qBX, qBY, rBX, rBY
        """
        self.model.eval()

        print("\n生成查询哈希码...")
        qBX = self._encode_images(self.query_images)
        qBY = self._encode_tags(self.query_tags)

        print("生成数据库哈希码...")
        rBX = self._encode_images(self.database_images)
        rBY = self._encode_tags(self.database_tags)

        return qBX, qBY, rBX, rBY

    @torch.no_grad()
    def _encode_images(self, images: torch.Tensor) -> np.ndarray:
        """编码图像。"""
        self.model.eval()

        n_samples = len(images)
        n_batches = (n_samples + self.batch_size - 1) // self.batch_size

        hash_codes = []

        for i in range(n_batches):
            start_idx = i * self.batch_size
            end_idx = min((i + 1) * self.batch_size, n_samples)

            batch_images = images[start_idx:end_idx] / 255.0

            if self.use_gpu:
                batch_images = batch_images.cuda()

            output = self.model.image_module(batch_images)
            binary_hash = torch.sign(output).cpu().numpy()

            hash_codes.append(binary_hash)

        return np.vstack(hash_codes)

    @torch.no_grad()
    def _encode_tags(self, tags: torch.Tensor) -> np.ndarray:
        """编码文本标签。"""
        self.model.eval()

        n_samples = len(tags)
        n_batches = (n_samples + self.batch_size - 1) // self.batch_size

        hash_codes = []

        for i in range(n_batches):
            start_idx = i * self.batch_size
            end_idx = min((i + 1) * self.batch_size, n_samples)

            batch_tags = tags[start_idx:end_idx].unsqueeze(1).unsqueeze(-1)

            if self.use_gpu:
                batch_tags = batch_tags.cuda()

            output = self.model.text_module(batch_tags)
            binary_hash = torch.sign(output).cpu().numpy()

            hash_codes.append(binary_hash)

        return np.vstack(hash_codes)

    def evaluate_map(self,
                    qBX: np.ndarray,
                    qBY: np.ndarray,
                    rBX: np.ndarray,
                    rBY: np.ndarray) -> Dict[str, float]:
        """
        评估 mAP。

        参数:
            qBX: 查询图像哈希码 (N, bit)
            qBY: 查询文本哈希码 (N, bit)
            rBX: 数据库图像哈希码 (M, bit)
            rBY: 数据库文本哈希码 (M, bit)

        返回:
            mAP 结果字典
        """
        print("\n" + "=" * 60)
        print("评估 mAP")
        print("=" * 60)

        qBX_tensor = torch.from_numpy(qBX).float()
        qBY_tensor = torch.from_numpy(qBY).float()
        rBX_tensor = torch.from_numpy(rBX).float()
        rBY_tensor = torch.from_numpy(rBY).float()

        query_labels = self.query_labels.cpu().numpy()
        database_labels = self.database_labels.cpu().numpy()

        # 图像->文本检索
        print("图像 -> 文本检索...")
        map_i2t = calc_map_k(
            qBX_tensor, rBY_tensor,
            torch.from_numpy(query_labels),
            torch.from_numpy(database_labels)
        )

        # 文本->图像检索
        print("文本 -> 图像检索...")
        map_t2i = calc_map_k(
            qBY_tensor, rBX_tensor,
            torch.from_numpy(query_labels),
            torch.from_numpy(database_labels)
        )

        results = {
            'map_i2t': map_i2t.item(),
            'map_t2i': map_t2i.item(),
            'map_avg': ((map_i2t + map_t2i) / 2).item()
        }

        print(f"\nMAP(i->t): {results['map_i2t']:.4f}")
        print(f"MAP(t->i): {results['map_t2i']:.4f}")
        print(f"平均 MAP: {results['map_avg']:.4f}")

        return results

    def evaluate_precision_recall(self,
                                 qBX: np.ndarray,
                                 qBY: np.ndarray,
                                 rBX: np.ndarray,
                                 rBY: np.ndarray,
                                 k_values: List[int] = [1, 5, 10, 20, 50]) -> Dict:
        """
        评估 Precision@K 和 Recall@K。

        返回:
            结果字典
        """
        print("\n" + "=" * 60)
        print("评估 Precision@K 和 Recall@K")
        print("=" * 60)

        # 计算相似度矩阵
        sim_i2t = np.dot(qBX, rBY.T)
        sim_t2i = np.dot(qBY, rBX.T)

        query_labels = np.argmax(self.query_labels.cpu().numpy(), axis=1)
        database_labels = np.argmax(self.database_labels.cpu().numpy(), axis=1)

        results = {
            'image_to_text': {'precision': {}, 'recall': {}},
            'text_to_image': {'precision': {}, 'recall': {}}
        }

        # 图像->文本
        print("图像 -> 文本...")
        retrieved_indices_i2t = np.argsort(-sim_i2t, axis=1)

        for k in k_values:
            if k <= retrieved_indices_i2t.shape[1]:
                p = compute_precision_at_k(
                    query_labels, retrieved_indices_i2t, database_labels, k
                )
                r = compute_recall_at_k(
                    query_labels, retrieved_indices_i2t, database_labels, k
                )
                results['image_to_text']['precision'][f'p@{k}'] = p
                results['image_to_text']['recall'][f'r@{k}'] = r
                print(f"  P@{k}: {p:.4f}, R@{k}: {r:.4f}")

        # 文本->图像
        print("文本 -> 图像...")
        retrieved_indices_t2i = np.argsort(-sim_t2i, axis=1)

        for k in k_values:
            if k <= retrieved_indices_t2i.shape[1]:
                p = compute_precision_at_k(
                    query_labels, retrieved_indices_t2i, database_labels, k
                )
                r = compute_recall_at_k(
                    query_labels, retrieved_indices_t2i, database_labels, k
                )
                results['text_to_image']['precision'][f'p@{k}'] = p
                results['text_to_image']['recall'][f'r@{k}'] = r
                print(f"  P@{k}: {p:.4f}, R@{k}: {r:.4f}")

        return results

    def evaluate_hash_quality(self,
                             hash_codes: np.ndarray) -> Dict[str, float]:
        """
        评估哈希码质量。

        返回:
            质量指标字典
        """
        print("\n" + "=" * 60)
        print("评估哈希码质量")
        print("=" * 60)

        quality = compute_hash_quality(hash_codes)

        print(f"平衡性：{quality['balance']:.4f}")
        print(f"唯一性：{quality['uniqueness']:.4f}")
        print(f"平均汉明距离：{quality['avg_hamming_distance']:.4f}")
        print(f"稀疏性：{quality['sparsity']:.4f}")

        return quality

    def generate_visualizations(self,
                               map_results: Dict,
                               pr_results: Dict,
                               quality_results: Dict,
                               training_history: Dict = None) -> List[str]:
        """
        生成可视化图表。

        返回:
            图表文件路径列表
        """
        print("\n" + "=" * 60)
        print("生成可视化图表")
        print("=" * 60)

        chart_paths = []

        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

        # 1. MAP 对比图
        print("生成 MAP 对比图...")
        fig, ax = plt.subplots(figsize=(10, 6))

        categories = ['图像→文本', '文本→图像', '平均']
        values = [
            map_results['map_i2t'],
            map_results['map_t2i'],
            map_results['map_avg']
        ]

        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        bars = ax.bar(categories, values, color=colors, alpha=0.8)

        ax.set_ylabel('mAP')
        ax.set_title(f'DCMH 检索性能 (bit={self.bit_dim})')
        ax.set_ylim(0, 1)

        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                   f'{val:.4f}', ha='center', va='bottom', fontsize=11)

        plt.tight_layout()
        map_chart_path = self.result_dir / 'evaluation_map.png'
        plt.savefig(map_chart_path, dpi=150, bbox_inches='tight')
        plt.close()
        chart_paths.append(str(map_chart_path))
        print(f"  已保存：{map_chart_path}")

        # 2. Precision@K 对比图
        print("生成 Precision@K 对比图...")
        fig, ax = plt.subplots(figsize=(12, 6))

        k_labels = list(pr_results['image_to_text']['precision'].keys())
        p_i2t = [pr_results['image_to_text']['precision'][k] for k in k_labels]
        p_t2i = [pr_results['text_to_image']['precision'][k] for k in k_labels]

        x = np.arange(len(k_labels))
        width = 0.35

        ax.bar(x - width/2, p_i2t, width, label='图像→文本', color='#1f77b4', alpha=0.8)
        ax.bar(x + width/2, p_t2i, width, label='文本→图像', color='#ff7f0e', alpha=0.8)

        ax.set_xlabel('K')
        ax.set_ylabel('Precision')
        ax.set_title('Precision@K 对比')
        ax.set_xticks(x)
        ax.set_xticklabels([k.replace('@', '') for k in k_labels])
        ax.legend()
        ax.set_ylim(0, 1)
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        precision_chart_path = self.result_dir / 'evaluation_precision.png'
        plt.savefig(precision_chart_path, dpi=150, bbox_inches='tight')
        plt.close()
        chart_paths.append(str(precision_chart_path))
        print(f"  已保存：{precision_chart_path}")

        # 3. Recall@K 对比图
        print("生成 Recall@K 对比图...")
        fig, ax = plt.subplots(figsize=(12, 6))

        r_i2t = [pr_results['image_to_text']['recall'][k] for k in k_labels]
        r_t2i = [pr_results['text_to_image']['recall'][k] for k in k_labels]

        x = np.arange(len(k_labels))
        width = 0.35

        ax.bar(x - width/2, r_i2t, width, label='图像→文本', color='#1f77b4', alpha=0.8)
        ax.bar(x + width/2, r_t2i, width, label='文本→图像', color='#ff7f0e', alpha=0.8)

        ax.set_xlabel('K')
        ax.set_ylabel('Recall')
        ax.set_title('Recall@K 对比')
        ax.set_xticks(x)
        ax.set_xticklabels([k.replace('@', '') for k in k_labels])
        ax.legend()
        ax.set_ylim(0, 1)
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        recall_chart_path = self.result_dir / 'evaluation_recall.png'
        plt.savefig(recall_chart_path, dpi=150, bbox_inches='tight')
        plt.close()
        chart_paths.append(str(recall_chart_path))
        print(f"  已保存：{recall_chart_path}")

        # 4. 哈希码质量雷达图
        print("生成哈希码质量雷达图...")
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))

        categories = ['平衡性', '唯一性', '平均汉明距离', '稀疏性']
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]

        values = [
            quality_results['balance'],
            quality_results['uniqueness'],
            min(quality_results['avg_hamming_distance'] / 0.5, 1.0),
            1.0 - abs(quality_results['sparsity'])
        ]
        values += values[:1]

        ax.plot(angles, values, 'o-', linewidth=2, color='#1f77b4')
        ax.fill(angles, values, alpha=0.25, color='#1f77b4')

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 1)
        ax.set_title('哈希码质量雷达图')
        ax.grid(True)

        plt.tight_layout()
        quality_chart_path = self.result_dir / 'evaluation_hash_quality.png'
        plt.savefig(quality_chart_path, dpi=150, bbox_inches='tight')
        plt.close()
        chart_paths.append(str(quality_chart_path))
        print(f"  已保存：{quality_chart_path}")

        # 5. 如果有训练历史，绘制训练曲线
        if training_history:
            print("生成训练曲线图...")
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            # 损失曲线
            ax1 = axes[0]
            epochs = range(1, len(training_history['loss']) + 1)
            ax1.plot(epochs, training_history['loss'], 'b-', linewidth=1.5)
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.set_title('训练损失曲线')
            ax1.grid(True, alpha=0.3)

            # mAP 曲线
            if 'map_i2t' in training_history and training_history['map_i2t']:
                ax2 = axes[1]
                map_epochs = range(
                    training_history.get('valid_interval', 10),
                    len(training_history['loss']) + 1,
                    training_history.get('valid_interval', 10)
                )
                ax2.plot(map_epochs, training_history['map_i2t'],
                        'b-', label='MAP(i->t)', linewidth=1.5)
                ax2.plot(map_epochs, training_history['map_t2i'],
                        'r-', label='MAP(t->i)', linewidth=1.5)
                ax2.plot(map_epochs, training_history['map_avg'],
                        'g-', label='平均 MAP', linewidth=2)
                ax2.set_xlabel('Epoch')
                ax2.set_ylabel('mAP')
                ax2.set_title('mAP 变化曲线')
                ax2.legend()
                ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            training_chart_path = self.result_dir / 'evaluation_training_curve.png'
            plt.savefig(training_chart_path, dpi=150, bbox_inches='tight')
            plt.close()
            chart_paths.append(str(training_chart_path))
            print(f"  已保存：{training_chart_path}")

        return chart_paths

    def generate_report(self,
                       map_results: Dict,
                       pr_results: Dict,
                       quality_results: Dict,
                       chart_paths: List[str]) -> str:
        """
        生成 Markdown 评估报告。

        返回:
            报告文件路径
        """
        print("\n" + "=" * 60)
        print("生成评估报告")
        print("=" * 60)

        report_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        report_content = f"""# DCMH Flickr25K 评估报告

**生成时间**: {report_date}
**数据集**: Flickr25K
**哈希码维度**: {self.bit_dim} bits
**模型**: {self.model_path}

---

## 1. 评估概述

本次评估测试了 DCMH 模型在 Flickr25K 数据集上的跨模态检索性能。

### 1.1 数据集统计

| 划分 | 样本数 |
|------|--------|
| 查询集 | {len(self.query_images):,} |
| 数据库 | {len(self.database_images):,} |
| 类别数 | {self.n_classes} |
| 文本维度 | {self.y_dim} |

### 1.2 评估指标

- **mAP**: 平均精度均值，衡量检索整体性能
- **Precision@K**: 前 K 个结果的精确率
- **Recall@K**: 前 K 个结果的召回率
- **哈希质量**: 平衡性、唯一性、汉明距离

---

## 2. mAP 评估结果

| 检索方向 | mAP |
|----------|-----|
| 图像 → 文本 | {map_results['map_i2t']:.4f} |
| 文本 → 图像 | {map_results['map_t2i']:.4f} |
| **平均** | **{map_results['map_avg']:.4f}** |

---

## 3. Precision@K 和 Recall@K

### 3.1 图像 → 文本检索

| K | Precision | Recall |
|---|-----------|--------|
"""

        for k in pr_results['image_to_text']['precision'].keys():
            p = pr_results['image_to_text']['precision'][k]
            r = pr_results['image_to_text']['recall'][k]
            report_content += f"| {k.replace('@', '')} | {p:.4f} | {r:.4f} |\n"

        report_content += f"""
### 3.2 文本 → 图像检索

| K | Precision | Recall |
|---|-----------|--------|
"""

        for k in pr_results['text_to_image']['precision'].keys():
            p = pr_results['text_to_image']['precision'][k]
            r = pr_results['text_to_image']['recall'][k]
            report_content += f"| {k.replace('@', '')} | {p:.4f} | {r:.4f} |\n"

        report_content += f"""
---

## 4. 哈希码质量

| 指标 | 值 |
|------|-----|
| 平衡性 | {quality_results['balance']:.4f} |
| 唯一性 | {quality_results['uniqueness']:.4f} |
| 平均汉明距离 | {quality_results['avg_hamming_distance']:.4f} |
| 稀疏性 | {quality_results['sparsity']:.4f} |

---

## 5. 可视化图表

"""

        chart_captions = {
            'evaluation_map.png': '图 1: mAP 对比',
            'evaluation_precision.png': '图 2: Precision@K 对比',
            'evaluation_recall.png': '图 3: Recall@K 对比',
            'evaluation_hash_quality.png': '图 4: 哈希码质量雷达图',
            'evaluation_training_curve.png': '图 5: 训练曲线'
        }

        for chart_path in chart_paths:
            chart_name = Path(chart_path).name
            caption = chart_captions.get(chart_name, chart_name)
            report_content += f"### {caption}\n\n"
            report_content += f"![{caption}]({chart_name})\n\n"

        report_content += f"""
---

## 6. 结论

基于评估结果：

1. **检索性能**:
   - 图像→文本 mAP: {map_results['map_i2t']:.4f}
   - 文本→图像 mAP: {map_results['map_t2i']:.4f}
   - 平均 mAP: **{map_results['map_avg']:.4f}**

2. **哈希码质量**:
   - 平衡性：{quality_results['balance']:.4f} (接近 1.0 为理想)
   - 唯一性：{quality_results['uniqueness']:.4f} (接近 1.0 为理想)

3. **推荐应用**:
   - 适用于大规模跨模态检索场景
   - 支持隐私保护的加密检索

---

**评估完成** | 详细数据请查看 `evaluation_results.json`
"""

        # 写入文件
        report_path = self.result_dir / 'evaluation_report.md'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)

        print(f"  已保存：{report_path}")

        return str(report_path)

    def evaluate(self, training_history: Dict = None) -> Dict:
        """
        执行完整评估。

        参数:
            training_history: 可选的训练历史

        返回:
            所有评估结果
        """
        print("\n" + "=" * 60)
        print("开始执行评估")
        print(f"时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)

        # 1. 生成哈希码
        qBX, qBY, rBX, rBY = self.generate_hash_codes()

        # 2. 评估 mAP
        map_results = self.evaluate_map(qBX, qBY, rBX, rBY)

        # 3. 评估 Precision@K 和 Recall@K
        pr_results = self.evaluate_precision_recall(
            qBX, qBY, rBX, rBY,
            k_values=[1, 5, 10, 20, 50, 100]
        )

        # 4. 评估哈希质量
        quality_results = self.evaluate_hash_quality(qBX)

        # 5. 生成可视化
        chart_paths = self.generate_visualizations(
            map_results, pr_results, quality_results,
            training_history
        )

        # 6. 生成报告
        report_path = self.generate_report(
            map_results, pr_results, quality_results,
            chart_paths
        )

        # 7. 保存 JSON 结果
        all_results = {
            'config': {
                'bit_dim': self.bit_dim,
                'model_path': str(self.model_path),
                'data_path': str(self.data_path),
                'evaluated_at': datetime.now().isoformat()
            },
            'map_results': map_results,
            'precision_recall_results': pr_results,
            'hash_quality_results': quality_results,
            'chart_paths': chart_paths,
            'report_path': report_path
        }

        json_path = self.result_dir / 'evaluation_results.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

        print(f"\n  JSON 结果已保存：{json_path}")

        # 总结
        print("\n" + "=" * 60)
        print("评估完成")
        print("=" * 60)
        print(f"平均 mAP: {map_results['map_avg']:.4f}")
        print(f"图表数量：{len(chart_paths)}")
        print(f"报告路径：{report_path}")

        return all_results


def evaluate_flickr25k(data_path: str,
                      model_path: str,
                      result_dir: str = 'results/flickr-25k',
                      **kwargs) -> Dict:
    """
    评估 Flickr25K 的便捷函数。

    参数:
        data_path: 数据文件路径
        model_path: 模型文件路径
        result_dir: 结果目录
        **kwargs: 传递给评估器的参数

    返回:
        评估结果字典
    """
    evaluator = Flickr25KEvaluator(
        data_path=data_path,
        model_path=model_path,
        result_dir=result_dir,
        **kwargs
    )

    return evaluator.evaluate()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='DCMH Flickr25K 评估')
    parser.add_argument('--data', type=str, default='data/flickr25k/FLICKR-25K.mat',
                       help='数据文件路径')
    parser.add_argument('--model', type=str, required=True,
                       help='模型文件路径')
    parser.add_argument('--result-dir', type=str, default='results/flickr-25k',
                       help='结果目录')
    parser.add_argument('--bit', type=int, default=64, help='哈希码维度')
    parser.add_argument('--no-gpu', action='store_true', help='禁用 GPU')

    args = parser.parse_args()

    results = evaluate_flickr25k(
        data_path=args.data,
        model_path=args.model,
        result_dir=args.result_dir,
        bit_dim=args.bit,
        use_gpu=not args.no_gpu
    )

    print("\n评估完成！")
