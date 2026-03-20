"""
构建加密数据库示例

使用ASPE从预训练的深度哈希模型特征构建加密数据库的示例脚本。
"""

import os
import pickle
import numpy as np
import torch

from core.hashing.dual_stream import DualStreamHashModel
from core.retrieval.pipeline import RetrievalPipeline
from data.coco_dataset import COCORetrievalDataset, get_coco_transforms
from config.model_config import DUAL_STREAM_CONFIG
from config.aspe_config import ASPE_SCHEME1_CONFIG, ASPE_SCHEME2_CONFIG


def load_model(checkpoint_path: str, device: str = 'cpu'):
    """加载预训练模型。"""
    model = DualStreamHashModel(
        vocab_size=DUAL_STREAM_CONFIG['text']['vocab_size'],
        feature_dim=DUAL_STREAM_CONFIG['feature_dim'],
        hash_bits=DUAL_STREAM_CONFIG['hash_bits'],
        image_backbone=DUAL_STREAM_CONFIG['image']['backbone'],
        text_encoder_type=DUAL_STREAM_CONFIG['text'].get('encoder_type', 'transformer'),
        pretrained=False  # 从检查点加载
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model


def extract_features_from_dataset(model, dataset, device: str = 'cpu'):
    """从数据集中提取特征。"""
    features_list = []
    metadata_list = []

    model.eval()
    with torch.no_grad():
        for batch in dataset:
            images = batch['image'].to(device)
            image_ids = batch['image_id']

            # 提取特征
            features = model.get_features('image', images)
            features_list.append(features.cpu().numpy())
            metadata_list.extend(image_ids.tolist())

    return np.concatenate(features_list, axis=0), metadata_list


def main():
    """构建加密数据库的主函数。"""
    print("=" * 60)
    print("构建加密数据库")
    print("=" * 60)

    # 配置
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    aspe_scheme = 'scheme1'  # 或 'scheme2'

    print(f"\n设备: {device}")
    print(f"ASPE方案: {aspe_scheme}")

    # 加载模型
    checkpoint_path = './checkpoints/best_model.pth'
    if not os.path.exists(checkpoint_path):
        print(f"\n错误: 在 {checkpoint_path} 未找到检查点")
        print("请先使用 train_on_coco.py 训练模型")
        return

    print(f"\n从 {checkpoint_path} 加载模型...")
    model = load_model(checkpoint_path, device)
    print("模型加载成功")

    # 加载数据集
    print("\n加载数据集...")
    from config.coco_config import COCO_CONFIG

    data_root = COCO_CONFIG['data_root']
    val_images = os.path.join(data_root, COCO_CONFIG['val_split'])
    val_annotations = os.path.join(data_root, 'annotations',
                                    COCO_CONFIG['val_split'] + '.json')

    transform = get_coco_transforms(image_size=224, is_training=False)

    # 创建图像数据集
    image_dataset = COCORetrievalDataset(
        image_dir=val_images,
        annotations_file=val_annotations,
        transform=transform,
        split='val',
        use_all_captions=False
    )

    print(f"数据集大小: {len(image_dataset)} 张图像")

    # 提取特征
    print("\n提取图像特征...")
    image_features, image_metadata = extract_features_from_dataset(
        model, image_dataset, device
    )

    print(f"提取的特征形状: {image_features.shape}")

    # 对于文本，您需要从字幕提取特征
    # 这是一个占位符 - 在实际应用中，需要分词并提取文本特征
    print("\n注意: 文本特征提取将在此处进行")
    print("(需要分词后的字幕)")

    # 使用ASPE创建检索管道
    print(f"\n使用 {aspe_scheme} 创建加密数据库...")
    pipeline = RetrievalPipeline(
        hash_model=model,
        aspe_scheme=aspe_scheme,
        device=device
    )

    # 构建加密数据库
    stats = pipeline.prepare_database(
        images=image_dataset,
        image_metadata=image_metadata,
        batch_size=32
    )

    print("\n加密数据库统计:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # 保存加密数据库
    output_dir = './results'
    os.makedirs(output_dir, exist_ok=True)

    db_path = os.path.join(output_dir, f'encrypted_db_{aspe_scheme}.pkl')
    pipeline.save_database(db_path)

    print(f"\n加密数据库已保存至: {db_path}")

    # 保存原始特征以供比较
    features_path = os.path.join(output_dir, 'image_features.npy')
    np.save(features_path, image_features)
    print(f"原始特征已保存至: {features_path}")

    metadata_path = os.path.join(output_dir, 'image_metadata.pkl')
    with open(metadata_path, 'wb') as f:
        pickle.dump(image_metadata, f)
    print(f"元数据已保存至: {metadata_path}")

    print("\n" + "=" * 60)
    print("加密数据库构建成功!")
    print("=" * 60)


if __name__ == "__main__":
    main()
