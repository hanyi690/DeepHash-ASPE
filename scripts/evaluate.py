#!/usr/bin/env python
"""
DCMH 统一评估命令行入口

提供便捷的命令行接口用于评估 DCMH 模型。

使用示例：
    # 基本评估
    python scripts/evaluate.py --result_dir results/flickr-25k/20260324_090727

    # 指定数据集和哈希码维度
    python scripts/evaluate.py --result_dir results/flickr-25k/20260324_090727 \\
        --data data/flickr25k/FLICKR-25K.mat --bit 64

    # 使用 CPU 评估
    python scripts/evaluate.py --result_dir results/flickr-25k/20260324_090727 --no-gpu

    # 跳过 ASPE 加密评估
    python scripts/evaluate.py --result_dir results/flickr-25k/20260324_090727 --no-aspe
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.evaluator import DCMHEvaluator
from evaluation.visualization import plot_training_history_from_json


def main():
    parser = argparse.ArgumentParser(
        description='DCMH 统一评估工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基本评估
  %(prog)s --result_dir results/flickr-25k/20260324_090727

  # 指定数据集和哈希码维度
  %(prog)s --result_dir results/flickr-25k/20260324_090727 --data data/flickr25k/FLICKR-25K.mat --bit 64

  # 使用 CPU 评估
  %(prog)s --result_dir results/flickr-25k/20260324_090727 --no-gpu

  # 跳过 ASPE 加密评估
  %(prog)s --result_dir results/flickr-25k/20260324_090727 --no-aspe

输出文件：
  - eval_map.png: mAP 对比图
  - eval_precision.png: Precision@K 对比图
  - eval_recall.png: Recall@K 对比图
  - eval_hash_quality.png: 哈希码质量雷达图
  - aspe_comparison.png: ASPE 明文 vs 密文对比图（默认启用）
  - training_loss.png: 训练损失曲线（如 result.json 存在）
  - training_map.png: mAP 训练曲线（如 result.json 存在）
  - evaluation_report.md: 评估报告
  - eval_results.json: 完整评估结果
        """
    )

    parser.add_argument('--result_dir', type=str, required=True,
                       help='训练结果目录（包含模型文件）')
    parser.add_argument('--data', type=str,
                       default='data/flickr25k/FLICKR-25K.mat',
                       help='数据文件路径 (默认: data/flickr25k/FLICKR-25K.mat)')
    parser.add_argument('--bit', type=int, default=64,
                       help='哈希码维度 (默认: 64)')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='批次大小 (默认: 128)')
    parser.add_argument('--no-gpu', action='store_true',
                       help='禁用 GPU，使用 CPU 评估')
    parser.add_argument('--no-aspe', action='store_true',
                       help='跳过 ASPE 加密评估')

    args = parser.parse_args()

    # 验证目录存在
    if not os.path.exists(args.result_dir):
        print(f"错误：结果目录不存在：{args.result_dir}")
        sys.exit(1)

    # 验证数据文件存在
    if not os.path.exists(args.data):
        print(f"警告：数据文件不存在：{args.data}")
        print("请确保数据文件路径正确，或使用 --data 参数指定正确的路径。")

    print("=" * 60)
    print("DCMH 统一评估工具")
    print("=" * 60)
    print(f"时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"结果目录：{args.result_dir}")
    print(f"数据文件：{args.data}")
    print(f"哈希码维度：{args.bit}")
    print(f"批次大小：{args.batch_size}")
    print(f"使用 GPU：{not args.no_gpu}")
    print(f"ASPE 评估：{not args.no_aspe}")
    print("=" * 60)

    # 生成训练曲线（如果 result.json 存在）
    result_json = os.path.join(args.result_dir, 'result.json')
    if os.path.exists(result_json):
        print("\n从 result.json 生成训练曲线...")
        plot_training_history_from_json(result_json, args.result_dir)

    try:
        evaluator = DCMHEvaluator(
            result_dir=args.result_dir,
            data_path=args.data,
            bit=args.bit,
            batch_size=args.batch_size,
            use_gpu=not args.no_gpu
        )

        results = evaluator.evaluate(run_aspe=not args.no_aspe)

        print("\n" + "=" * 60)
        print("评估完成！")
        print("=" * 60)
        print(f"\n输出目录：{args.result_dir}")
        print("\n生成的文件：")
        for path in results.get('chart_paths', []):
            print(f"  - {path}")
        print(f"  - {results.get('report_path', 'evaluation_report.md')}")
        print(f"  - eval_results.json")

        print(f"\n平均 mAP: {results['map_results']['map_avg']:.4f}")

        # 打印 ASPE 评估结果
        if results.get('aspe_results'):
            aspe = results['aspe_results']
            print("\nASPE 加密评估结果：")
            print(f"  明文 mAP(i→t): {aspe['plaintext']['map_i2t']:.6f}")
            print(f"  密文 mAP(i→t): {aspe['ciphertext']['map_i2t']:.6f}")
            print(f"  明文 mAP(t→i): {aspe['plaintext']['map_t2i']:.6f}")
            print(f"  密文 mAP(t→i): {aspe['ciphertext']['map_t2i']:.6f}")
            print(f"  排序一致性验证: {'通过' if aspe['consistency_verified'] else '失败'}")

    except FileNotFoundError as e:
        print(f"\n错误：{e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n评估失败：{e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()