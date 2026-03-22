"""
Flickr25K 快速实验脚本

简化版的 Flickr25K 实验运行脚本，适合快速测试和验证。
"""

import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from experiments.run_flickr25k import run_flickr25k_experiment


def main():
    """运行快速实验。"""
    print("=" * 60)
    print("Flickr25K 快速实验")
    print("=" * 60)

    # 简化配置
    results = run_flickr25k_experiment(
        data_path='data/flickr25k/FLICKR-25K.mat',
        bit_dim=64,
        max_epoch=100,  # 快速测试用较少轮数
        batch_size=64,
        lr=1e-4,
        result_dir='results/flickr-25k-quick',
        use_gpu=False  # 默认不使用 GPU
    )

    print("\n" + "=" * 60)
    print("快速实验完成！")
    print("=" * 60)
    print(f"结果目录：results/flickr-25k-quick")
    print(f"最佳 MAP: {results['training']['best_map']:.4f}")


if __name__ == "__main__":
    main()
