#!/usr/bin/env python
"""
CNN 图像检索演示数据库构建脚本

构建用于演示的加密检索数据库，包括：
1. 下载预训练模型
2. 提取图像特征
3. 使用 SkNN 加密特征
4. 保存加密数据库
"""

import os
import sys
import argparse
from pathlib import Path

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.cirtorch.services.sknn_service import SknnService


def build_demo_database(
    image_dir: str,
    save_dir: str,
    model_path: str = None,
    feature_dim: int = 2048,
    batch_size: int = 32
):
    """
    构建演示用 CNN 检索数据库。

    参数:
        image_dir: 图像目录
        save_dir: 数据库保存目录
        model_path: 预训练模型路径
        feature_dim: 特征维度
        batch_size: 批次大小
    """
    image_dir = Path(image_dir)
    save_dir = Path(save_dir)

    # 检查图像目录
    if not image_dir.exists():
        print(f"[错误] 图像目录不存在: {image_dir}")
        return False

    # 检查模型
    if model_path is None:
        model_path = PROJECT_ROOT / 'data' / 'models' / 'resnet101-gem.pth'
        if not model_path.exists():
            print(f"[警告] 默认模型不存在: {model_path}")
            print("[提示] 请先运行: python scripts/download_cir_model.py")
            return False

    print(f"\n{'='*60}")
    print("构建 CNN 检索演示数据库")
    print(f"{'='*60}")
    print(f"图像目录: {image_dir}")
    print(f"保存目录: {save_dir}")
    print(f"模型路径: {model_path}")
    print(f"特征维度: {feature_dim}")
    print(f"{'='*60}\n")

    # 创建服务
    print("[1/4] 初始化 SkNN 服务...")
    service = SknnService(
        feature_dim=feature_dim,
        model_path=str(model_path),
        db_dir=str(save_dir)
    )

    # 加载模型
    print("[2/4] 加载检索模型...")
    service.load_model(str(model_path))

    # 生成密钥
    print("[3/4] 生成 SkNN 密钥...")
    service.generate_keys()

    # 构建数据库
    print("[4/4] 构建加密数据库...")
    db_features, db_images = service.build_database(
        image_dir=str(image_dir),
        save_dir=str(save_dir),
        batch_size=batch_size
    )

    print(f"\n{'='*60}")
    print("数据库构建完成！")
    print(f"{'='*60}")
    print(f"数据库大小: {len(db_images)} 张图像")
    print(f"特征形状: {db_features.shape}")
    print(f"保存路径: {save_dir}")
    print(f"\n生成的文件:")
    print(f"  - encrypted_features.pth  (加密特征)")
    print(f"  - image_names.pth         (图像名称)")
    print(f"  - sknn_keys.pth           (SkNN 密钥)")
    print(f"{'='*60}\n")

    return True


def verify_database(db_dir: str):
    """
    验证数据库完整性。

    参数:
        db_dir: 数据库目录
    """
    import torch

    db_dir = Path(db_dir)

    print(f"\n{'='*60}")
    print("验证数据库")
    print(f"{'='*60}\n")

    files = ['encrypted_features.pth', 'image_names.pth', 'sknn_keys.pth']

    for f in files:
        path = db_dir / f
        if path.exists():
            print(f"[OK] {f}")
        else:
            print(f"[缺失] {f}")
            return False

    # 加载并检查
    try:
        features = torch.load(db_dir / 'encrypted_features.pth', weights_only=False)
        names = torch.load(db_dir / 'image_names.pth', weights_only=False)
        keys = torch.load(db_dir / 'sknn_keys.pth', weights_only=False)

        print(f"\n数据库信息:")
        print(f"  图像数量: {len(names)}")
        print(f"  特征形状: {features.shape}")
        print(f"  密钥 M1 形状: {keys['M1'].shape}")
        print(f"  密钥 M2 形状: {keys['M2'].shape}")

        print(f"\n[OK] 数据库验证通过")
        return True

    except Exception as e:
        print(f"\n[错误] 数据库验证失败: {e}")
        return False


def count_images(image_dir: str) -> int:
    """统计图像数量。"""
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    count = 0
    for f in Path(image_dir).iterdir():
        if f.suffix.lower() in valid_extensions:
            count += 1
    return count


def main():
    parser = argparse.ArgumentParser(
        description='CNN 图像检索演示数据库构建工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 构建演示数据库
  python build_cir_demo_db.py --image-dir data/flickr-25k/images --save-dir data/cir_demo_db

  # 指定模型路径
  python build_cir_demo_db.py --image-dir ./images --save-dir ./db --model ./model.pth

  # 验证数据库
  python build_cir_demo_db.py --verify data/cir_demo_db
        """
    )
    parser.add_argument(
        '--image-dir', '-i',
        type=str,
        help='图像目录路径'
    )
    parser.add_argument(
        '--save-dir', '-s',
        type=str,
        default='data/cir_demo_db',
        help='数据库保存目录 (默认: data/cir_demo_db)'
    )
    parser.add_argument(
        '--model', '-m',
        type=str,
        default=None,
        help='预训练模型路径 (默认: data/models/resnet101-gem.pth)'
    )
    parser.add_argument(
        '--feature-dim', '-d',
        type=int,
        default=2048,
        help='特征维度 (默认: 2048)'
    )
    parser.add_argument(
        '--batch-size', '-b',
        type=int,
        default=32,
        help='批次大小 (默认: 32)'
    )
    parser.add_argument(
        '--verify', '-v',
        type=str,
        help='验证指定目录的数据库'
    )

    args = parser.parse_args()

    # 验证模式
    if args.verify:
        verify_database(args.verify)
        return

    # 构建模式
    if not args.image_dir:
        parser.error("请指定 --image-dir 参数")

    # 检查图像数量
    num_images = count_images(args.image_dir)
    print(f"\n发现 {num_images} 张图像")

    if num_images == 0:
        print(f"[错误] 图像目录为空: {args.image_dir}")
        return

    # 构建数据库
    success = build_demo_database(
        image_dir=args.image_dir,
        save_dir=args.save_dir,
        model_path=args.model,
        feature_dim=args.feature_dim,
        batch_size=args.batch_size
    )

    if success:
        # 验证结果
        verify_database(args.save_dir)


if __name__ == '__main__':
    main()