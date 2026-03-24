"""
CIR (CNN Image Retrieval) 数据库构建脚本

功能：
1. 加载预训练 CNN 检索模型
2. 批量提取图像特征
3. 使用 ASPE 加密特征向量
4. 保存加密数据库和密钥

用法：
    python scripts/build_cir_db.py --image_dir <图像目录> --save_dir <保存目录>
"""

import os
import sys
import argparse
import torch
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.app.services.cir_service import CIRService


def main():
    parser = argparse.ArgumentParser(description='构建 CIR 加密图像检索数据库')

    parser.add_argument('--image_dir', type=str, required=True,
                       help='图像数据库目录')
    parser.add_argument('--save_dir', type=str, default='./data/retrieval_db',
                       help='加密数据库保存目录')
    parser.add_argument('--model_path', type=str,
                       default='./data/networks/gl18-tl-resnet101-gem-w-a4d43db.pth',
                       help='预训练模型路径')
    parser.add_argument('--feature_dim', type=int, default=2048,
                       help='特征维度（默认 2048 对应 ResNet101-GeM）')
    parser.add_argument('--device', type=str, default=None,
                       help='计算设备（默认自动选择）')

    args = parser.parse_args()

    print("=" * 60)
    print("CIR 加密图像检索数据库构建工具")
    print("=" * 60)

    # 初始化服务
    service = CIRService(
        feature_dim=args.feature_dim,
        model_path=args.model_path,
        db_dir=args.save_dir,
        device=args.device
    )

    # 构建数据库（自动完成模型加载、密钥生成、特征加密）
    db_features, db_images = service.build_database(
        image_dir=args.image_dir,
        save_dir=args.save_dir
    )

    print("\n" + "=" * 60)
    print("建库完成！")
    print("=" * 60)
    print(f"密文库维度：{db_features.shape}")
    print(f"图像数量：{len(db_images)}")
    print(f"保存路径：{args.save_dir}")
    print("\n生成的文件:")
    print(f"  - encrypted_features.pth: 加密特征矩阵")
    print(f"  - image_names.pth: 图像名称列表")
    print(f"  - aspe_keys.pth: ASPE 加密密钥")


if __name__ == "__main__":
    main()