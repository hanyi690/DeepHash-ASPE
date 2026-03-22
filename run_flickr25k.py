"""
Flickr25K 完整系统测试和实验运行器

提供两种模式：
1. 完整实验：训练 500 轮，生成完整报告和图表
2. 快速测试：训练 50 轮，快速验证系统功能
"""

import sys
import argparse
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))


def run_experiment(mode: str = 'full', **kwargs):
    """
    运行实验。

    参数：
        mode: 'full' 完整实验 或 'quick' 快速测试
        **kwargs: 覆盖默认配置
    """
    from experiments.run_flickr25k import Flickr25KExperiment

    # 默认配置
    if mode == 'quick':
        config = {
            'data_path': 'data/flickr25k/FLICKR-25K.mat',
            'bit_dim': 64,
            'max_epoch': 50,
            'batch_size': 64,
            'lr': 1e-4,
            'gamma': 1.0,
            'eta': 1.0,
            'use_gpu': False,
            'result_dir': 'results/flickr-25k-quick'
        }
        print("=" * 60)
        print("Flickr25K 快速测试模式")
        print("=" * 60)
        print("注意：快速测试仅训练 50 轮，用于验证系统功能")
        print("完整实验请使用：python run_flickr25k.py --mode full")
    else:
        config = {
            'data_path': 'data/flickr25k/FLICKR-25K.mat',
            'bit_dim': 64,
            'max_epoch': 500,
            'batch_size': 128,
            'lr': 1e-4,
            'gamma': 1.0,
            'eta': 1.0,
            'use_gpu': True,
            'result_dir': 'results/flickr-25k'
        }
        print("=" * 60)
        print("Flickr25K 完整实验模式")
        print("=" * 60)

    # 应用覆盖配置
    config.update(kwargs)

    print(f"\n配置:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    # 创建实验
    experiment = Flickr25KExperiment(**config)

    # 运行实验
    results = experiment.run_full_experiment()

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Flickr25K 完整系统测试和实验',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 快速测试（50 轮，约 10-30 分钟）
  python run_flickr25k.py --mode quick

  # 完整实验（500 轮，约需数小时）
  python run_flickr25k.py --mode full

  # 自定义配置
  python run_flickr25k.py --bit 32 --epochs 200 --batch-size 64
        """
    )

    parser.add_argument(
        '--mode',
        type=str,
        default='full',
        choices=['full', 'quick'],
        help='实验模式：full (完整) 或 quick (快速)'
    )
    parser.add_argument(
        '--data',
        type=str,
        default='data/flickr25k/FLICKR-25K.mat',
        help='数据文件路径'
    )
    parser.add_argument(
        '--bit',
        type=int,
        default=None,
        help='哈希码维度 (默认：64)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='最大训练轮数'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='批次大小'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=None,
        help='学习率'
    )
    parser.add_argument(
        '--result-dir',
        type=str,
        default=None,
        help='结果输出目录'
    )
    parser.add_argument(
        '--no-gpu',
        action='store_true',
        help='禁用 GPU'
    )

    args = parser.parse_args()

    # 构建配置
    config = {'mode': args.mode}

    if args.data:
        config['data_path'] = args.data
    if args.bit:
        config['bit_dim'] = args.bit
    if args.epochs:
        config['max_epoch'] = args.epochs
    if args.batch_size:
        config['batch_size'] = args.batch_size
    if args.lr:
        config['lr'] = args.lr
    if args.result_dir:
        config['result_dir'] = args.result_dir
    if args.no_gpu:
        config['use_gpu'] = False

    # 运行实验
    results = run_experiment(**config)

    # 打印摘要
    print("\n" + "=" * 60)
    print("实验摘要")
    print("=" * 60)

    if results:
        print(f"训练轮数：{len(results['training']['history']['loss'])}")
        print(f"最佳 MAP: {results['training']['best_map']:.4f} (Epoch {results['training']['best_epoch']})")
        print(f"评估 MAP(i->t): {results['evaluation']['map']['i2t']:.4f}")
        print(f"评估 MAP(t->i): {results['evaluation']['map']['t2i']:.4f}")
        print(f"平均 MAP: {results['evaluation']['map']['avg']:.4f}")

        if results['aspe_test']['map_preservation']['passed']:
            print("ASPE mAP 保持性：✓ 通过")
        else:
            print("ASPE mAP 保持性：✗ 未通过")

    print(f"\n结果目录：{config.get('result_dir', 'results/flickr-25k')}")
    print("=" * 60)


if __name__ == "__main__":
    main()
