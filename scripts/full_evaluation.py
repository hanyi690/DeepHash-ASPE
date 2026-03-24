"""
DCMH 完整评估脚本

评估训练好的 DCMH 模型，生成完整的评估报告和可视化图表。
"""

import os
import sys
import json
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.dcmh_config import DCMHConfig
from core.hashing.dcmh_data_loader import load_data, split_data
from core.hashing.dcmh_image import build_dcmh_image_model
from core.hashing.dcmh_text import build_dcmh_text_model
from core.retrieval.dcmh_metrics import calc_map_k
from evaluation.metrics import (
    compute_precision_at_k,
    compute_recall_at_k,
    compute_hash_quality
)


class DCMHEvaluator:
    """DCMH 完整评估器。"""

    def __init__(self,
                 result_dir: str,
                 data_path: str = 'data/flickr25k/FLICKR-25K.mat',
                 bit: int = 64,
                 batch_size: int = 128,
                 use_gpu: bool = True):
        """
        初始化评估器。

        参数:
            result_dir: 训练结果目录（包含 img_model.pth 和 txt_model.pth）
            data_path: 数据文件路径
            bit: 哈希码维度
            batch_size: 批次大小
            use_gpu: 是否使用 GPU
        """
        self.result_dir = Path(result_dir)
        self.data_path = data_path
        self.bit = bit
        self.batch_size = batch_size
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = torch.device('cuda' if self.use_gpu else 'cpu')

        # 模型路径
        self.img_model_path = self.result_dir / 'img_model.pth'
        self.txt_model_path = self.result_dir / 'txt_model.pth'

        # 加载数据和模型
        self._load_data()
        self._load_model()

    def _load_data(self):
        """加载数据。"""
        print("=" * 60)
        print("加载 Flickr25K 数据")
        print("=" * 60)

        images, tags, labels = load_data(self.data_path)
        X, Y, L = split_data(images, tags, labels,
                            query_size=2000,
                            training_size=10000,
                            database_size=18015)

        self.query_images = torch.from_numpy(X['query']).float()
        self.query_tags = torch.from_numpy(Y['query']).float()
        self.query_labels = torch.from_numpy(L['query']).float()

        self.database_images = torch.from_numpy(X['retrieval']).float()
        self.database_tags = torch.from_numpy(Y['retrieval']).float()
        self.database_labels = torch.from_numpy(L['retrieval']).float()

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

        self.img_model = build_dcmh_image_model(bit=self.bit, pretrain_model_path=None)
        self.txt_model = build_dcmh_text_model(y_dim=self.y_dim, bit=self.bit)

        print(f"加载图像模型：{self.img_model_path}")
        self.img_model.load_state_dict(
            torch.load(self.img_model_path, map_location=self.device)
        )

        print(f"加载文本模型：{self.txt_model_path}")
        self.txt_model.load_state_dict(
            torch.load(self.txt_model_path, map_location=self.device)
        )

        if self.use_gpu:
            self.img_model = self.img_model.cuda()
            self.txt_model = self.txt_model.cuda()

        print(f"模型加载完成！设备：{self.device}")

    @torch.no_grad()
    def generate_hash_codes(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """生成哈希码。"""
        print("\n生成查询图像哈希码...")
        qBX = self._encode_images(self.query_images)

        print("生成查询文本哈希码...")
        qBY = self._encode_tags(self.query_tags)

        print("生成数据库图像哈希码...")
        rBX = self._encode_images(self.database_images)

        print("生成数据库文本哈希码...")
        rBY = self._encode_tags(self.database_tags)

        return qBX, qBY, rBX, rBY

    @torch.no_grad()
    def _encode_images(self, images: torch.Tensor) -> np.ndarray:
        """编码图像。"""
        self.img_model.eval()
        n_samples = len(images)
        hash_codes = []

        for i in range(0, n_samples, self.batch_size):
            batch = images[i:i+self.batch_size]
            if self.use_gpu:
                batch = batch.cuda()
            output = self.img_model(batch)
            binary = torch.sign(output).cpu().numpy()
            hash_codes.append(binary)

        return np.vstack(hash_codes)

    @torch.no_grad()
    def _encode_tags(self, tags: torch.Tensor) -> np.ndarray:
        """编码文本标签。"""
        self.txt_model.eval()
        n_samples = len(tags)
        hash_codes = []

        for i in range(0, n_samples, self.batch_size):
            batch = tags[i:i+self.batch_size].unsqueeze(1).unsqueeze(-1)
            if self.use_gpu:
                batch = batch.cuda()
            output = self.txt_model(batch)
            binary = torch.sign(output).cpu().numpy()
            hash_codes.append(binary)

        return np.vstack(hash_codes)

    def evaluate_map(self, qBX, qBY, rBX, rBY) -> Dict[str, float]:
        """计算 mAP。"""
        print("\n" + "=" * 60)
        print("计算 mAP")
        print("=" * 60)

        query_labels = self.query_labels
        database_labels = self.database_labels

        if self.use_gpu:
            query_labels = query_labels.cuda()
            database_labels = database_labels.cuda()

        qBX_t = torch.from_numpy(qBX).float()
        qBY_t = torch.from_numpy(qBY).float()
        rBX_t = torch.from_numpy(rBX).float()
        rBY_t = torch.from_numpy(rBY).float()

        if self.use_gpu:
            qBX_t = qBX_t.cuda()
            qBY_t = qBY_t.cuda()
            rBX_t = rBX_t.cuda()
            rBY_t = rBY_t.cuda()

        print("图像 -> 文本检索...")
        map_i2t = calc_map_k(qBX_t, rBY_t, query_labels, database_labels)

        print("文本 -> 图像检索...")
        map_t2i = calc_map_k(qBY_t, rBX_t, query_labels, database_labels)

        results = {
            'map_i2t': float(map_i2t),
            'map_t2i': float(map_t2i),
            'map_avg': float((map_i2t + map_t2i) / 2)
        }

        print(f"\nMAP(i->t): {results['map_i2t']:.4f}")
        print(f"MAP(t->i): {results['map_t2i']:.4f}")
        print(f"平均 MAP: {results['map_avg']:.4f}")

        return results

    def evaluate_precision_recall(self, qBX, qBY, rBX, rBY,
                                  k_values: List[int] = [1, 5, 10, 20, 50, 100]) -> Dict:
        """计算 Precision@K 和 Recall@K。"""
        print("\n" + "=" * 60)
        print("计算 Precision@K 和 Recall@K")
        print("=" * 60)

        # 计算相似度
        sim_i2t = np.dot(qBX, rBY.T)
        sim_t2i = np.dot(qBY, rBX.T)

        # 转换标签为类别索引
        query_labels = np.argmax(self.query_labels.numpy(), axis=1)
        database_labels = np.argmax(self.database_labels.numpy(), axis=1)

        results = {
            'image_to_text': {'precision': {}, 'recall': {}},
            'text_to_image': {'precision': {}, 'recall': {}}
        }

        # 图像 -> 文本
        print("\n图像 -> 文本:")
        retrieved_i2t = np.argsort(-sim_i2t, axis=1)
        for k in k_values:
            p = compute_precision_at_k(query_labels, retrieved_i2t, database_labels, k)
            r = compute_recall_at_k(query_labels, retrieved_i2t, database_labels, k)
            results['image_to_text']['precision'][f'p@{k}'] = p
            results['image_to_text']['recall'][f'r@{k}'] = r
            print(f"  P@{k}: {p:.4f}, R@{k}: {r:.4f}")

        # 文本 -> 图像
        print("\n文本 -> 图像:")
        retrieved_t2i = np.argsort(-sim_t2i, axis=1)
        for k in k_values:
            p = compute_precision_at_k(query_labels, retrieved_t2i, database_labels, k)
            r = compute_recall_at_k(query_labels, retrieved_t2i, database_labels, k)
            results['text_to_image']['precision'][f'p@{k}'] = p
            results['text_to_image']['recall'][f'r@{k}'] = r
            print(f"  P@{k}: {p:.4f}, R@{k}: {r:.4f}")

        return results

    def evaluate_hash_quality(self, hash_codes: np.ndarray) -> Dict[str, float]:
        """评估哈希码质量。"""
        print("\n" + "=" * 60)
        print("评估哈希码质量")
        print("=" * 60)

        quality = compute_hash_quality(hash_codes)

        print(f"平衡性：{quality['balance']:.4f}")
        print(f"唯一性：{quality['uniqueness']:.4f}")
        print(f"平均汉明距离：{quality['avg_hamming_distance']:.4f}")
        print(f"稀疏性：{quality['sparsity']:.4f}")

        return quality

    def generate_visualizations(self, map_results: Dict, pr_results: Dict,
                               quality_results: Dict) -> List[str]:
        """生成可视化图表。"""
        print("\n" + "=" * 60)
        print("生成可视化图表")
        print("=" * 60)

        chart_paths = []

        # 设置字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

        # 1. MAP 对比图
        print("生成 MAP 对比图...")
        fig, ax = plt.subplots(figsize=(10, 6))

        categories = ['图像→文本', '文本→图像', '平均']
        values = [map_results['map_i2t'], map_results['map_t2i'], map_results['map_avg']]
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

        bars = ax.bar(categories, values, color=colors, alpha=0.8)
        ax.set_ylabel('mAP')
        ax.set_title(f'DCMH 检索性能 (bit={self.bit})')
        ax.set_ylim(0, 1)

        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                   f'{val:.4f}', ha='center', va='bottom', fontsize=11)

        plt.tight_layout()
        path = self.result_dir / 'eval_map.png'
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        chart_paths.append(str(path))
        print(f"  已保存：{path}")

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
        path = self.result_dir / 'eval_precision.png'
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        chart_paths.append(str(path))
        print(f"  已保存：{path}")

        # 3. Recall@K 对比图
        print("生成 Recall@K 对比图...")
        fig, ax = plt.subplots(figsize=(12, 6))

        # 获取 recall 键名
        r_labels = list(pr_results['image_to_text']['recall'].keys())
        r_i2t = [pr_results['image_to_text']['recall'][k] for k in r_labels]
        r_t2i = [pr_results['text_to_image']['recall'][k] for k in r_labels]

        x_r = np.arange(len(r_labels))

        ax.bar(x_r - width/2, r_i2t, width, label='图像→文本', color='#1f77b4', alpha=0.8)
        ax.bar(x_r + width/2, r_t2i, width, label='文本→图像', color='#ff7f0e', alpha=0.8)

        ax.set_xlabel('K')
        ax.set_ylabel('Recall')
        ax.set_title('Recall@K 对比')
        ax.set_xticks(x_r)
        ax.set_xticklabels([k.replace('@', '') for k in r_labels])
        ax.legend()
        ax.set_ylim(0, 1)
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        path = self.result_dir / 'eval_recall.png'
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        chart_paths.append(str(path))
        print(f"  已保存：{path}")

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
        path = self.result_dir / 'eval_hash_quality.png'
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        chart_paths.append(str(path))
        print(f"  已保存：{path}")

        return chart_paths

    def generate_report(self, map_results: Dict, pr_results: Dict,
                       quality_results: Dict, chart_paths: List[str]) -> str:
        """生成 Markdown 评估报告。"""
        print("\n" + "=" * 60)
        print("生成评估报告")
        print("=" * 60)

        report_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        report = f"""# DCMH Flickr25K 评估报告

**生成时间**: {report_date}
**数据集**: Flickr25K
**哈希码维度**: {self.bit} bits
**模型目录**: {self.result_dir}

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

        for pk in pr_results['image_to_text']['precision'].keys():
            p = pr_results['image_to_text']['precision'][pk]
            # 将 p@1 转换为 r@1
            rk = pk.replace('p@', 'r@')
            r = pr_results['image_to_text']['recall'][rk]
            report += f"| {pk.replace('@', '')} | {p:.4f} | {r:.4f} |\n"

        report += f"""
### 3.2 文本 → 图像检索

| K | Precision | Recall |
|---|-----------|--------|
"""

        for pk in pr_results['text_to_image']['precision'].keys():
            p = pr_results['text_to_image']['precision'][pk]
            # 将 p@1 转换为 r@1
            rk = pk.replace('p@', 'r@')
            r = pr_results['text_to_image']['recall'][rk]
            report += f"| {pk.replace('@', '')} | {p:.4f} | {r:.4f} |\n"

        report += f"""
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

        captions = {
            'eval_map.png': '图 1: mAP 对比',
            'eval_precision.png': '图 2: Precision@K 对比',
            'eval_recall.png': '图 3: Recall@K 对比',
            'eval_hash_quality.png': '图 4: 哈希码质量雷达图'
        }

        for chart_path in chart_paths:
            chart_name = Path(chart_path).name
            caption = captions.get(chart_name, chart_name)
            report += f"### {caption}\n\n![{caption}]({chart_name})\n\n"

        report += f"""
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

**评估完成** | 详细数据请查看 `eval_results.json`
"""

        # 写入文件
        report_path = self.result_dir / 'evaluation_report.md'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)

        print(f"  已保存：{report_path}")

        return str(report_path)

    def evaluate(self) -> Dict:
        """执行完整评估。"""
        print("\n" + "=" * 60)
        print("DCMH 完整评估")
        print(f"时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)

        # 1. 生成哈希码
        qBX, qBY, rBX, rBY = self.generate_hash_codes()

        # 2. 计算 mAP
        map_results = self.evaluate_map(qBX, qBY, rBX, rBY)

        # 3. 计算 Precision@K 和 Recall@K
        pr_results = self.evaluate_precision_recall(qBX, qBY, rBX, rBY)

        # 4. 评估哈希质量
        quality_results = self.evaluate_hash_quality(qBX)

        # 5. 生成可视化
        chart_paths = self.generate_visualizations(map_results, pr_results, quality_results)

        # 6. 生成报告
        report_path = self.generate_report(map_results, pr_results, quality_results, chart_paths)

        # 7. 保存 JSON 结果
        def convert_to_serializable(obj):
            """将 numpy 类型转换为 Python 类型"""
            if isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_to_serializable(v) for v in obj]
            elif isinstance(obj, (np.floating, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.integer, np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        all_results = {
            'config': {
                'bit': self.bit,
                'result_dir': str(self.result_dir),
                'evaluated_at': datetime.now().isoformat()
            },
            'map_results': map_results,
            'precision_recall': pr_results,
            'hash_quality': quality_results,
            'chart_paths': chart_paths,
            'report_path': report_path
        }

        json_path = self.result_dir / 'eval_results.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(convert_to_serializable(all_results), f, indent=2, ensure_ascii=False)

        print(f"\n  JSON 结果已保存：{json_path}")

        # 总结
        print("\n" + "=" * 60)
        print("评估完成")
        print("=" * 60)
        print(f"平均 mAP: {map_results['map_avg']:.4f}")
        print(f"图表数量：{len(chart_paths)}")
        print(f"报告路径：{report_path}")

        return all_results


def main():
    import argparse

    parser = argparse.ArgumentParser(description='DCMH 完整评估')
    parser.add_argument('--result_dir', type=str, required=True,
                       help='训练结果目录')
    parser.add_argument('--data', type=str, default='data/flickr25k/FLICKR-25K.mat',
                       help='数据文件路径')
    parser.add_argument('--bit', type=int, default=64, help='哈希码维度')
    parser.add_argument('--no-gpu', action='store_true', help='禁用 GPU')

    args = parser.parse_args()

    evaluator = DCMHEvaluator(
        result_dir=args.result_dir,
        data_path=args.data,
        bit=args.bit,
        use_gpu=not args.no_gpu
    )

    results = evaluator.evaluate()

    print("\n评估完成！")


if __name__ == "__main__":
    main()