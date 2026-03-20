"""
图像→文本检索示例

演示使用加密数据库进行图像到文本的检索。
"""

import os
import numpy as np
import torch
from PIL import Image

from core.hashing.dual_stream import DualStreamHashModel
from core.retrieval.pipeline import RetrievalPipeline
from config.model_config import DUAL_STREAM_CONFIG


def load_and_preprocess_image(image_path: str):
    """加载并预处理图像。"""
    image = Image.open(image_path).convert('RGB')
    image = image.resize((224, 224))

    # 转换为张量并标准化
    import torchvision.transforms as transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    return transform(image)


def main():
    """主图像到文本检索示例。"""
    print("=" * 60)
    print("图像→文本检索示例")
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
    print(f"  文本数: {db_size['text']}")

    # 示例查询图像
    print("\n" + "=" * 60)
    print("示例图像查询")
    print("=" * 60)

    # 在实际应用中，您需要提供实际的图像路径
    # 对于此示例，我们将使用随机张量
    print("\n注意: 在实际应用中，提供实际的图像路径")
    print("对于此示例，使用随机图像张量\n")

    for i in range(3):
        print(f"查询图像 {i + 1}:")

        # 生成随机查询图像
        query_image = torch.randn(3, 224, 224)

        # 执行检索
        results = pipeline.image_to_text(query_image, k=5)

        print(f"\n前5个检索文本:")
        for j, result in enumerate(results[:5], 1):
            print(f"  {j}. {result['metadata']} - 分数: {result['score']:.4f}")

        print()

    # 使用实际图像测试（如果可用）
    print("\n" + "=" * 60)
    print("使用自定义图像测试")
    print("=" * 60)

    # 提供图像路径
    custom_image_path = input("\n输入查询图像的路径（或按Enter跳过）: ").strip()

    if custom_image_path and os.path.exists(custom_image_path):
        print(f"\n从 {custom_image_path} 加载图像...")

        try:
            query_image = load_and_preprocess_image(custom_image_path)

            # 执行检索
            results = pipeline.image_to_text(query_image, k=10)

            print(f"\n前10个检索文本:")
            for j, result in enumerate(results[:10], 1):
                print(f"  {j}. {result['metadata']} - 分数: {result['score']:.4f}")

        except Exception as e:
            print(f"处理图像时出错: {e}")
    else:
        print("跳过自定义图像查询")

    # 基准测试查询时间
    print("\n" + "=" * 60)
    print("性能基准测试")
    print("=" * 60)

    import time

    query_image = torch.randn(3, 224, 224)

    times = []
    for _ in range(10):
        start = time.perf_counter()
        _ = pipeline.image_to_text(query_image, k=10)
        end = time.perf_counter()
        times.append(end - start)

    avg_time = np.mean(times) * 1000
    std_time = np.std(times) * 1000

    print(f"\n查询时间 (10次运行, k=10):")
    print(f"  平均: {avg_time:.2f} 毫秒")
    print(f"  标准差: {std_time:.2f} 毫秒")

    print("\n" + "=" * 60)
    print("图像→文本检索完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
