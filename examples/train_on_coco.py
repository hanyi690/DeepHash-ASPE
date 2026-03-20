"""
在MS-COCO数据集上的训练示例

在MS-COCO 2014数据集上训练双流深度哈希模型的示例脚本。
"""

import os
import torch
from torch.utils.data import DataLoader

from core.hashing.dual_stream import DualStreamHashModel
from data.coco_dataset import COCOCrossModalDataset, get_coco_transforms, create_coco_dataloaders
from data.tokenizers import SimpleTokenizer, Vocabulary
from training.trainer import create_trainer
from config.train_config import TRAINING_CONFIG
from config.model_config import DUAL_STREAM_CONFIG
from config.coco_config import COCO_CONFIG


def main():
    """主训练函数。"""
    print("=" * 60)
    print("Deep Hash + ASPE 在 MS-COCO 上训练")
    print("=" * 60)

    # 设置设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n设备: {device}")

    # 路径（更新为您的COCO数据集路径）
    data_root = COCO_CONFIG['data_root']
    train_images = os.path.join(data_root, COCO_CONFIG['train_split'])
    train_annotations = os.path.join(data_root, 'annotations', COCO_CONFIG['train_split'] + '.json')
    val_images = os.path.join(data_root, COCO_CONFIG['val_split'])
    val_annotations = os.path.join(data_root, 'annotations', COCO_CONFIG['val_split'] + '.json')

    print(f"\n数据路径:")
    print(f"  训练图像: {train_images}")
    print(f"  训练标注: {train_annotations}")
    print(f"  验证图像: {val_images}")
    print(f"  验证标注: {val_annotations}")

    # 从字幕构建词汇表
    print("\n" + "=" * 60)
    print("构建词汇表...")
    print("=" * 60)

    # 为简单起见，此示例使用小型固定词汇表
    # 在实际应用中，您应该从训练字幕构建它
    vocab_size = DUAL_STREAM_CONFIG['text']['vocab_size']
    print(f"使用词汇表大小: {vocab_size}")

    # 创建变换
    train_transform = get_coco_transforms(image_size=224, is_training=True)
    val_transform = get_coco_transforms(image_size=224, is_training=False)

    # 创建数据集
    print("\n" + "=" * 60)
    print("创建数据集...")
    print("=" * 60)

    train_dataset = COCOCrossModalDataset(
        image_dir=train_images,
        annotations_file=train_annotations,
        transform=train_transform,
        max_caption_length=COCO_CONFIG['max_caption_length'],
        num_captions_per_image=COCO_CONFIG['num_captions_per_image']
    )

    print(f"训练样本数: {len(train_dataset)}")

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAINING_CONFIG['batch_size'],
        shuffle=True,
        num_workers=COCO_CONFIG['num_workers'],
        pin_memory=True
    )

    # 创建模型
    print("\n" + "=" * 60)
    print("创建模型...")
    print("=" * 60)

    model = DualStreamHashModel(
        vocab_size=vocab_size,
        feature_dim=DUAL_STREAM_CONFIG['feature_dim'],
        hash_bits=DUAL_STREAM_CONFIG['hash_bits'],
        image_backbone=DUAL_STREAM_CONFIG['image']['backbone'],
        text_encoder_type=DUAL_STREAM_CONFIG['text'].get('encoder_type', 'transformer'),
        pretrained=DUAL_STREAM_CONFIG['image']['pretrained'],
        dropout=DUAL_STREAM_CONFIG['dropout']
    )

    num_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数: {num_params:,}")

    # 创建训练器
    print("\n" + "=" * 60)
    print("创建训练器...")
    print("=" * 60)

    trainer = create_trainer(model, TRAINING_CONFIG, device)

    # 训练循环
    print("\n" + "=" * 60)
    print("开始训练...")
    print("=" * 60)

    history = trainer.train(
        train_loader=train_loader,
        val_loader=None,  # 如果需要，添加验证加载器
        num_epochs=TRAINING_CONFIG['num_epochs'],
        early_stopping_patience=TRAINING_CONFIG.get('patience'),
        save_frequency=TRAINING_CONFIG['save_frequency']
    )

    print("\n" + "=" * 60)
    print("训练完成!")
    print("=" * 60)

    # 保存最终模型
    final_checkpoint = os.path.join(TRAINING_CONFIG['checkpoint_dir'], 'final_model.pth')
    trainer.model.save_to(final_checkpoint)
    print(f"最终模型已保存至: {final_checkpoint}")


if __name__ == "__main__":
    # 注意：此脚本需要下载MS-COCO数据集。
    # 运行前请更新 config/coco_config.py 中的路径。

    try:
        main()
    except Exception as e:
        print(f"\n错误: {e}")
        print("\n请确保:")
        print("1. 已下载MS-COCO数据集")
        print("2. config/coco_config.py 中的路径正确")
        print("3. 所有依赖项已安装")
