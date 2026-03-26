"""
DCMH 统一评估器

提供统一的 DCMH 模型评估功能：
- 从训练结果目录加载模型
- 生成哈希码
- 计算 mAP、Precision@K、Recall@K
- 评估哈希码质量
- 生成可视化图表和报告
"""

import os
import sys
import json
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.dcmh_config import DCMHConfig
from core.hashing.dcmh_data_loader import load_data, split_data
from core.hashing.dcmh_image import build_dcmh_image_model
from core.hashing.dcmh_text import build_dcmh_text_model
from evaluation.metrics import (
    calc_map_k,
    compute_precision_at_k,
    compute_recall_at_k,
    compute_hash_quality
)
from evaluation.visualization import (
    plot_map_comparison,
    plot_precision_recall,
    plot_hash_quality_radar,
    plot_aspe_comparison,
    generate_evaluation_report
)


class DCMHEvaluator:
    """
    DCMH 统一评估器。

    支持两种模型加载方式：
    1. 从分离的 img_model.pth 和 txt_model.pth 加载（Reference 实现）
    2. 从单个 checkpoint 文件加载（统一模型）

    使用示例：
        # 方式 1：从结果目录评估
        evaluator = DCMHEvaluator(result_dir='results/flickr-25k/20260324_090727')
        results = evaluator.evaluate()

        # 方式 2：从模型和数据路径评估
        evaluator = DCMHEvaluator(
            result_dir='results/flickr-25k',
            data_path='data/flickr25k/FLICKR-25K.mat',
            bit=64
        )
        results = evaluator.evaluate()
    """

    def __init__(self,
                 result_dir: str,
                 data_path: str = 'data/flickr25k/FLICKR-25K.mat',
                 bit: int = 64,
                 batch_size: int = 128,
                 use_gpu: bool = True):
        """
        初始化评估器。

        参数：
            result_dir: 训练结果目录（包含模型文件）
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
        self.checkpoint_path = self.result_dir / 'checkpoint.pth'

        # 加载数据和模型
        self._load_data()
        self._load_model()

    def _load_data(self):
        """加载数据集。"""
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

        self.img_model = None
        self.txt_model = None
        self.model = None

        # 尝试加载分离模型（Reference 实现）
        if self.img_model_path.exists() and self.txt_model_path.exists():
            print(f"加载分离模型...")
            self.img_model = build_dcmh_image_model(bit=self.bit, pretrain_model_path=None)
            self.txt_model = build_dcmh_text_model(y_dim=self.y_dim, bit=self.bit)

            self.img_model.load_state_dict(
                torch.load(self.img_model_path, map_location=self.device)
            )
            self.txt_model.load_state_dict(
                torch.load(self.txt_model_path, map_location=self.device)
            )

            if self.use_gpu:
                self.img_model = self.img_model.cuda()
                self.txt_model = self.txt_model.cuda()

            print(f"  图像模型：{self.img_model_path}")
            print(f"  文本模型：{self.txt_model_path}")

        # 尝试加载统一模型
        elif self.checkpoint_path.exists():
            print(f"加载统一模型：{self.checkpoint_path}")
            from core.hashing.dcmh_model import DCMHModel
            self.model = DCMHModel(bit=self.bit, y_dim=self.y_dim)

            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])

            if self.use_gpu:
                self.model = self.model.cuda()

            print(f"  最佳 MAP: {checkpoint.get('best_map', 0):.4f}")

        else:
            raise FileNotFoundError(
                f"未找到模型文件。请确保目录中存在：\n"
                f"  - {self.img_model_path} 和 {self.txt_model_path}，或\n"
                f"  - {self.checkpoint_path}"
            )

        print(f"模型加载完成！设备：{self.device}")

    @torch.no_grad()
    def generate_hash_codes(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        生成查询和数据库的哈希码。

        返回：
            qBX, qBY, rBX, rBY 四个哈希码数组
        """
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
        """编码图像为哈希码。"""
        n_samples = len(images)
        hash_codes = []

        # 使用分离模型
        if self.img_model is not None:
            self.img_model.eval()
            for i in range(0, n_samples, self.batch_size):
                batch = images[i:i+self.batch_size]
                if self.use_gpu:
                    batch = batch.cuda()
                output = self.img_model(batch)
                binary = torch.sign(output).cpu().numpy()
                hash_codes.append(binary)

        # 使用统一模型
        elif self.model is not None:
            self.model.eval()
            for i in range(0, n_samples, self.batch_size):
                batch = images[i:i+self.batch_size]
                if self.use_gpu:
                    batch = batch.cuda()
                output = self.model.image_module(batch)
                binary = torch.sign(output).cpu().numpy()
                hash_codes.append(binary)

        return np.vstack(hash_codes)

    @torch.no_grad()
    def _encode_tags(self, tags: torch.Tensor) -> np.ndarray:
        """编码文本标签为哈希码。"""
        n_samples = len(tags)
        hash_codes = []

        # 使用分离模型
        if self.txt_model is not None:
            self.txt_model.eval()
            for i in range(0, n_samples, self.batch_size):
                batch = tags[i:i+self.batch_size].unsqueeze(1).unsqueeze(-1)
                if self.use_gpu:
                    batch = batch.cuda()
                output = self.txt_model(batch)
                binary = torch.sign(output).cpu().numpy()
                hash_codes.append(binary)

        # 使用统一模型
        elif self.model is not None:
            self.model.eval()
            for i in range(0, n_samples, self.batch_size):
                batch = tags[i:i+self.batch_size].unsqueeze(1).unsqueeze(-1)
                if self.use_gpu:
                    batch = batch.cuda()
                output = self.model.text_module(batch)
                binary = torch.sign(output).cpu().numpy()
                hash_codes.append(binary)

        return np.vstack(hash_codes)

    def evaluate_map(self, qBX, qBY, rBX, rBY) -> Dict[str, float]:
        """
        计算 mAP。

        参数：
            qBX: 查询图像哈希码
            qBY: 查询文本哈希码
            rBX: 数据库图像哈希码
            rBY: 数据库文本哈希码

        返回：
            包含 map_i2t, map_t2i, map_avg 的字典
        """
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
                                  k_values: List[int] = None) -> Dict:
        """
        计算 Precision@K 和 Recall@K。

        参数：
            qBX, qBY, rBX, rBY: 哈希码
            k_values: K 值列表

        返回：
            结果字典
        """
        if k_values is None:
            k_values = [1, 5, 10, 20, 50, 100]

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
        """
        评估哈希码质量。

        参数：
            hash_codes: 哈希码数组

        返回：
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

    def evaluate_aspe(self, qBX: np.ndarray, qBY: np.ndarray,
                      rBX: np.ndarray, rBY: np.ndarray,
                      map_results: Dict[str, float]) -> Dict[str, Any]:
        """
        执行 ASPE 加密评估。

        验证密文 mAP 与明文 mAP 是否相等。
        使用 Scheme 2（双矩阵增强方案）进行加密。

        参数：
            qBX: 查询图像哈希码
            qBY: 查询文本哈希码
            rBX: 数据库图像哈希码
            rBY: 数据库文本哈希码
            map_results: 明文 mAP 结果

        返回：
            ASPE 评估结果字典
        """
        from core.aspe.dcmh_wrapper_v2 import ASPEForDCMHv2

        print("\n" + "=" * 60)
        print("ASPE 加密评估 (Scheme 2)")
        print("=" * 60)

        # 1. 初始化 ASPE V2
        print(f"初始化 ASPE V2 (bit={self.bit}, Scheme 2)...")
        aspe = ASPEForDCMHv2(bit_dim=self.bit, seed=42)
        print(f"  扩展维度 d': {aspe.actual_d_prime}")
        print(f"  密文维度: {2 * aspe.actual_d_prime}")
        print(f"  安全级别: 3 (抵抗已知明文攻击)")

        # 2. 加密检索库哈希码
        print("加密检索库哈希码...")
        encrypted_rBX = aspe.GenEnc(rBX)
        encrypted_rBY = aspe.GenEnc(rBY)
        print(f"  加密后维度: {encrypted_rBX.shape}")

        # 3. 加密查询哈希码（生成陷阱门）
        print("生成查询陷阱门...")
        encrypted_qBX = aspe.GenTrap(qBX)
        encrypted_qBY = aspe.GenTrap(qBY)
        print(f"  加密后维度: {encrypted_qBX.shape}")

        # 4. 计算密文 mAP
        print("计算密文 mAP...")

        query_labels = self.query_labels.cpu().numpy()
        database_labels = self.database_labels.cpu().numpy()

        # 密文 mAP(i→t): 用图像查询文本
        print("  计算密文 mAP(i→t)...")
        cipher_map_i2t = aspe.calc_ciphertext_map(
            encrypted_qBX, encrypted_rBY,
            query_labels, database_labels
        )

        # 密文 mAP(t→i): 用文本查询图像
        print("  计算密文 mAP(t→i)...")
        cipher_map_t2i = aspe.calc_ciphertext_map(
            encrypted_qBY, encrypted_rBX,
            query_labels, database_labels
        )

        # 5. 验证排序一致性
        print("验证排序一致性...")
        consistent = aspe.verify_sorting_consistency(qBX[:10], rBX[:100])
        print(f"  排序一致性验证: {'通过' if consistent else '失败'}")

        # 6. 计算误差
        error_i2t = abs(cipher_map_i2t - map_results['map_i2t'])
        error_t2i = abs(cipher_map_t2i - map_results['map_t2i'])

        print("\n结果对比:")
        print(f"  mAP(i→t): 明文={map_results['map_i2t']:.6f}, 密文={cipher_map_i2t:.6f}, 误差={error_i2t:.2e}")
        print(f"  mAP(t→i): 明文={map_results['map_t2i']:.6f}, 密文={cipher_map_t2i:.6f}, 误差={error_t2i:.2e}")

        results = {
            'plaintext': {
                'map_i2t': float(map_results['map_i2t']),
                'map_t2i': float(map_results['map_t2i'])
            },
            'ciphertext': {
                'map_i2t': float(cipher_map_i2t),
                'map_t2i': float(cipher_map_t2i)
            },
            'error': {
                'map_i2t': float(error_i2t),
                'map_t2i': float(error_t2i)
            },
            'consistency_verified': consistent,
            'scheme': 'Scheme 2',
            'security_level': 3,
            'bit': self.bit,
            'd_prime': aspe.actual_d_prime,
            'ciphertext_dim': 2 * aspe.actual_d_prime
        }

        print("\nASPE 评估完成!")
        return results

    def generate_visualizations(self, map_results: Dict, pr_results: Dict,
                               quality_results: Dict,
                               aspe_results: Optional[Dict] = None) -> List[str]:
        """
        生成可视化图表。

        参数：
            map_results: mAP 结果
            pr_results: Precision@K / Recall@K 结果
            quality_results: 哈希码质量结果
            aspe_results: ASPE 评估结果（可选）

        返回：
            图表文件路径列表
        """
        print("\n" + "=" * 60)
        print("生成可视化图表")
        print("=" * 60)

        chart_paths = []

        # 1. MAP 对比图
        print("生成 MAP 对比图...")
        map_path = str(self.result_dir / 'eval_map.png')
        plot_map_comparison(map_results, map_path, bit=self.bit)
        chart_paths.append(map_path)
        print(f"  已保存：{map_path}")

        # 2. Precision@K / Recall@K 图
        print("生成 Precision@K 和 Recall@K 图...")
        pr_paths = plot_precision_recall(pr_results, str(self.result_dir), prefix='eval')
        chart_paths.extend(pr_paths.values())
        for name, path in pr_paths.items():
            print(f"  已保存：{path}")

        # 3. 哈希码质量雷达图
        print("生成哈希码质量雷达图...")
        quality_path = str(self.result_dir / 'eval_hash_quality.png')
        plot_hash_quality_radar(quality_results, quality_path)
        chart_paths.append(quality_path)
        print(f"  已保存：{quality_path}")

        # 4. ASPE 对比图（如果有 ASPE 结果）
        if aspe_results:
            print("生成 ASPE 明文 vs 密文对比图...")
            aspe_path = str(self.result_dir / 'aspe_comparison.png')
            plot_aspe_comparison(aspe_results, aspe_path)
            chart_paths.append(aspe_path)
            print(f"  已保存：{aspe_path}")

        return chart_paths

    def generate_report(self, map_results: Dict, pr_results: Dict,
                       quality_results: Dict, chart_paths: List[str],
                       aspe_results: Optional[Dict] = None) -> str:
        """
        生成 Markdown 评估报告。

        参数：
            map_results: mAP 结果
            pr_results: Precision@K / Recall@K 结果
            quality_results: 哈希码质量结果
            chart_paths: 图表路径列表
            aspe_results: ASPE 评估结果（可选）

        返回：
            报告文件路径
        """
        print("\n" + "=" * 60)
        print("生成评估报告")
        print("=" * 60)

        config = {
            'bit': self.bit,
            'result_dir': str(self.result_dir),
            'data_path': self.data_path
        }

        report_path = str(self.result_dir / 'evaluation_report.md')
        generate_evaluation_report(
            map_results, pr_results, quality_results,
            report_path, config, aspe_results
        )

        print(f"  已保存：{report_path}")
        return report_path

    def evaluate(self, run_aspe: bool = True) -> Dict[str, Any]:
        """
        执行完整评估。

        参数：
            run_aspe: 是否执行 ASPE 加密评估

        返回：
            所有评估结果
        """
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

        # 5. ASPE 加密评估
        aspe_results = None
        if run_aspe:
            aspe_results = self.evaluate_aspe(qBX, qBY, rBX, rBY, map_results)

        # 6. 生成可视化
        chart_paths = self.generate_visualizations(map_results, pr_results, quality_results, aspe_results)

        # 7. 生成报告
        report_path = self.generate_report(map_results, pr_results, quality_results, chart_paths, aspe_results)

        # 7. 保存 JSON 结果
        def convert_to_serializable(obj):
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
                'data_path': self.data_path,
                'evaluated_at': datetime.now().isoformat()
            },
            'map_results': map_results,
            'precision_recall': pr_results,
            'hash_quality': quality_results,
            'aspe_results': aspe_results,
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
    """命令行入口。"""
    import argparse

    parser = argparse.ArgumentParser(description='DCMH 统一评估器')
    parser.add_argument('--result_dir', type=str, required=True,
                       help='训练结果目录')
    parser.add_argument('--data', type=str, default='data/flickr25k/FLICKR-25K.mat',
                       help='数据文件路径')
    parser.add_argument('--bit', type=int, default=64, help='哈希码维度')
    parser.add_argument('--no-gpu', action='store_true', help='禁用 GPU')
    parser.add_argument('--no-aspe', action='store_true', help='跳过 ASPE 加密评估')

    args = parser.parse_args()

    evaluator = DCMHEvaluator(
        result_dir=args.result_dir,
        data_path=args.data,
        bit=args.bit,
        use_gpu=not args.no_gpu
    )

    results = evaluator.evaluate(run_aspe=not args.no_aspe)

    print("\n评估完成！")


if __name__ == "__main__":
    main()