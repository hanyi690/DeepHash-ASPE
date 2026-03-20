"""
文本→图像检索示例

演示使用加密数据库进行文本到图像的检索。
"""

import os
import pickle
import numpy as np
import torch

from core.hashing.dual_stream import DualStreamHashModel
from core.retrieval.pipeline import RetrievalPipeline
from config.model_config import DUAL_STREAM_CONFIG


def main():
    """主文本到图像检索示例。"""
    print("=" * 60)
    print("文本→图像检索示例")
    print("=" * 60)

    # 配置
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    aspe_scheme = 'scheme1'

    print(f"\n设备: {device}")
    print(f"ASPE方案: {aspe_scheme}")

    # 加载模型
    checkpoint_path = './checkpoints/best_model.pth'
    if not os.path.exists(checkpoint_path):
        print(f"\n错误: 未找到模型检查点")
        print("请先训练模型")
        return

    print(f"\n从 {checkpoint_path} 加载模型...")
    model = DualStreamHashModel(
        vocab_size=DUAL_STREAM_CONFIG['text']['vocab_size'],
        feature_dim=DUAL_STREAM_CONFIG['feature_dim'],
        hash_bits=DUAL_STREAM_CONFIG['hash_bits'],
        image_backbone=DUAL_STREAM_CONFIG['image']['backbone'],
        pretrained=False
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.to(device)

    print("模型加载成功")

    # 加载加密数据库
    db_path = f'./results/encrypted_db_{aspe_scheme}.pkl'
    if not os.path.exists(db_path):
        print(f"\n错误: 在 {db_path} 未找到加密数据库")
        print("请先使用 build_encrypted_db.py 构建数据库")
        return

    print(f"\n从 {db_path} 加载加密数据库...")
    pipeline = RetrievalPipeline(
        hash_model=model,
        aspe_scheme=aspe_scheme,
        device=device
    )
    pipeline.load_database(db_path)

    db_size = pipeline.engine.get_database_size()
    print(f"加密数据库已加载:")
    print(f"  图像数: {db_size['image']}")

    # 示例查询
    print("\n" + "=" * 60)
    print("示例文本查询")
    print("=" * 60)

    # 在实际应用中，您需要对这些文本查询进行分词
    # 对于此示例，我们将使用随机标记
    vocab_size = DUAL_STREAM_CONFIG['text']['vocab_size']
    max_seq_length = 50

    example_queries = [
        "A dog playing with a ball in the park",
        "Two cats sitting on a couch",
        "A person riding a bicycle",
        "Beautiful sunset over the ocean",
        "Group of people having dinner"
    ]

    print("\n注意: 在实际应用中，文本查询需要被分词")
    print("对于此示例，使用随机标记序列\n")

    for i, query_text in enumerate(example_queries, 1):
        print(f"查询 {i}: {query_text}")

        # 分词（占位符 - 在实际应用中使用实际分词）
        query_tokens = torch.randint(0, vocab_size, (1, max_seq_length))

        # 执行检索
        results = pipeline.text_to_image(query_tokens, k=5)

        print(f"\n前5个结果:")
        for j, result in enumerate(results[:5], 1):
            print(f"  {j}. {result['metadata']} - 分数: {result['score']:.4f}")

        print()

    # 基准测试查询时间
    print("\n" + "=" * 60)
    print("性能基准测试")
    print("=" * 60)

    import time

    query_tokens = torch.randint(0, vocab_size, (1, max_seq_length))

    times = []
    for _ in range(10):
        start = time.perf_counter()
        _ = pipeline.text_to_image(query_tokens, k=10)
        end = time.perf_counter()
        times.append(end - start)

    avg_time = np.mean(times) * 1000
    std_time = np.std(times) * 1000

    print(f"\n查询时间 (10次运行, k=10):")
    print(f"  平均: {avg_time:.2f} 毫秒")
    print(f"  标准差: {std_time:.2f} 毫秒")

    print("\n" + "=" * 60)
    print("文本→图像检索完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
