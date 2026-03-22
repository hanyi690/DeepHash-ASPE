"""
DCMH + ASPE 快速评估脚本

用于评估 DCMH 模型在 FLICKR-25K 数据集上的性能，
并验证 ASPE 加密前后 mAP 的一致性。

使用方式：
    python evaluation/quick_eval_dcmh_aspe.py --mode quick --bit 64

参数：
    --mode: 评估模式 (quick: 快速评估，full: 完整评估)
    --bit: 哈希码位数 (16, 32, 64, 128)
    --epochs: 训练轮数（仅 full 模式）
"""

import sys
import os
# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import time
import numpy as np
import torch
from pathlib import Path

from core.hashing.dcmh_image import DCMHImageModule
from core.hashing.dcmh_text import DCMHTextModule
from core.retrieval.dcmh_metrics import calc_map_k, calc_hammingDist
from core.aspe.dcmh_wrapper import ASPEForDCMH


def load_flickr25k(data_path):
    """
    加载 FLICKR-25K 数据集。

    参数：
        data_path: FLICKR-25K.mat 文件路径

    返回：
        images, tags, labels
    """
    import h5py
    file = h5py.File(data_path, 'r')
    images = file['images'][:].astype('float')
    labels = file['LAll'][:]
    tags = file['YAll'][:]
    file.close()
    return images, tags, labels


def prepare_data_split(images, tags, labels, training_size=10000, query_size=2000):
    """
    准备数据划分。

    将数据划分为训练集、查询集和数据库集。
    使用索引引用而非复制数据以节省内存。

    参数：
        images: 图像数据
        tags: 文本标签
        labels: 类别标签
        training_size: 训练集大小
        query_size: 查询集大小

    返回：
        train_idx, query_idx, database_idx (索引列表)
    """
    total_size = len(images)
    database_size = total_size - training_size - query_size

    # 简单划分：前 N 个为训练集，中间 M 个为查询集，其余为数据库
    train_idx = list(range(0, training_size))
    query_idx = list(range(training_size, training_size + query_size))
    database_idx = list(range(training_size + query_size, training_size + query_size + database_size))

    return train_idx, query_idx, database_idx


def preprocess_images(images):
    """
    预处理图像数据。

    将图像调整为 224x224 并转换为 PyTorch 张量。

    参数：
        images: 原始图像数据

    返回：
        预处理后的图像张量
    """
    from torchvision import transforms
    import torch.nn.functional as F

    processed = []
    for img in images:
        # 假设图像已经是 224x224x3 格式
        if img.shape[0] != 3:
            # 转换为 CHW 格式
            img = np.transpose(img, (2, 0, 1))

        # 调整为 224x224
        img_tensor = torch.from_numpy(img).float()
        if img_tensor.shape[1] != 224 or img_tensor.shape[2] != 224:
            img_tensor = F.interpolate(
                img_tensor.unsqueeze(0),
                size=(224, 224),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)

        processed.append(img_tensor)

    return torch.stack(processed)


def prepare_text_input(tags, y_dim):
    """
    准备文本输入。

    将标签索引转换为 one-hot/multi-hot 编码，并调整为卷积输入格式。

    参数：
        tags: 标签数组
        y_dim: 标签维度

    返回：
        文本输入张量 [batch, 1, y_dim, 1]
    """
    batch_size = len(tags)
    # 创建 multi-hot 编码
    one_hot = torch.zeros(batch_size, y_dim)
    for i, tag_indices in enumerate(tags):
        # 处理不同的标签格式
        if isinstance(tag_indices, np.ndarray):
            # 如果已经是数组，直接使用
            indices = tag_indices.flatten().astype(int)
        elif isinstance(tag_indices, (list, tuple)):
            indices = np.array(tag_indices).flatten().astype(int)
        else:
            # 单个值
            indices = np.array([int(tag_indices)])

        # 确保索引在有效范围内
        indices = indices[(indices >= 0) & (indices < y_dim)]
        if len(indices) > 0:
            one_hot[i, indices] = 1.0

    # 调整为卷积输入格式 [batch, 1, y_dim, 1]
    return one_hot.unsqueeze(1).unsqueeze(-1)


def generate_hash_codes(img_model, txt_model, data, bit, batch_size=64, use_gpu=True):
    """
    生成图像和文本哈希码。

    参数：
        img_model: 图像模型
        txt_model: 文本模型
        data: 数据字典（包含 images 和 tags）
        bit: 哈希码位数
        batch_size: 批次大小
        use_gpu: 是否使用 GPU

    返回：
        img_codes, txt_codes
    """
    num_data = len(data['images'])
    img_codes = torch.zeros(num_data, bit)
    txt_codes = torch.zeros(num_data, bit)

    if use_gpu:
        img_codes = img_codes.cuda()
        txt_codes = txt_codes.cuda()
        img_model = img_model.cuda()
        txt_model = txt_model.cuda()

    # 预处理图像
    print("预处理图像...")
    processed_images = preprocess_images(data['images'])

    # 准备文本输入
    print("准备文本输入...")
    y_dim = data['tags'].shape[1] if len(data['tags'].shape) > 1 else 1000
    text_input = prepare_text_input(data['tags'], y_dim)

    if use_gpu:
        processed_images = processed_images.cuda()
        text_input = text_input.cuda()

    img_model.eval()
    txt_model.eval()

    with torch.no_grad():
        # 生成图像哈希码
        print("生成图像哈希码...")
        for i in range(0, num_data, batch_size):
            end_idx = min(i + batch_size, num_data)
            img_batch = processed_images[i:end_idx]
            img_output = img_model(img_batch)
            img_codes[i:end_idx] = torch.sign(img_output)

        # 生成文本哈希码
        print("生成文本哈希码...")
        for i in range(0, num_data, batch_size):
            end_idx = min(i + batch_size, num_data)
            txt_batch = text_input[i:end_idx]
            txt_output = txt_model(txt_batch)
            txt_codes[i:end_idx] = torch.sign(txt_output)

    return img_codes.cpu(), txt_codes.cpu()


def quick_eval(bit=64, data_path='data/FLICKR-25K.mat', pretrain_path='checkpoints/imagenet-vgg-f.mat'):
    """
    快速评估：使用随机初始化的模型生成哈希码，不训练。

    参数：
        bit: 哈希码位数
        data_path: 数据集路径
        pretrain_path: 预训练模型路径
    """
    print(f"\n{'='*60}")
    print(f"DCMH + ASPE 快速评估 (bit={bit})")
    print(f"{'='*60}\n")

    # 加载数据
    print("加载 FLICKR-25K 数据集...")
    start_time = time.time()
    images, tags, labels = load_flickr25k(data_path)
    print(f"数据加载完成，耗时 {time.time() - start_time:.2f}s")
    print(f"  图像形状：{images.shape}")
    print(f"  标签形状：{tags.shape}")
    print(f"  类别形状：{labels.shape}")

    # 准备数据划分
    print("\n准备数据划分...")
    train_idx, query_idx, database_idx = prepare_data_split(images, tags, labels)
    print(f"  训练集：{len(train_idx)} 样本")
    print(f"  查询集：{len(query_idx)} 样本")
    print(f"  数据库：{len(database_idx)} 样本")

    # 初始化模型
    print(f"\n初始化 DCMH 模型...")
    y_dim = tags.shape[1] if len(tags.shape) > 1 else 1000

    # 检查预训练模型
    import os
    pretrain_data = None
    if os.path.exists(pretrain_path):
        print(f"加载预训练模型：{pretrain_path}")
        import scipy.io as scio
        pretrain_data = scio.loadmat(pretrain_path)
    else:
        print(f"警告：预训练模型不存在 {pretrain_path}，使用随机初始化")

    img_model = DCMHImageModule(bit=bit, pretrain_model=pretrain_data)
    txt_model = DCMHTextModule(y_dim=y_dim, bit=bit)

    use_gpu = torch.cuda.is_available()
    if use_gpu:
        img_model = img_model.cuda()
        txt_model = txt_model.cuda()
        print("使用 GPU 加速")
    else:
        print("使用 CPU 运行")

    # 生成哈希码（使用查询集和数据库集）
    print("\n生成哈希码（使用随机/预训练权重，不训练）...")
    start_time = time.time()

    # 为了节省内存，直接使用索引获取数据
    query_data = {
        'images': images[query_idx],
        'tags': tags[query_idx],
        'labels': labels[query_idx]
    }
    database_data = {
        'images': images[database_idx],
        'tags': tags[database_idx],
        'labels': labels[database_idx]
    }

    # 生成哈希码
    img_codes_query, txt_codes_query = generate_hash_codes(
        img_model, txt_model, query_data, bit, use_gpu=use_gpu
    )
    img_codes_db, txt_codes_db = generate_hash_codes(
        img_model, txt_model, database_data, bit, use_gpu=use_gpu
    )

    print(f"哈希码生成完成，耗时 {time.time() - start_time:.2f}s")

    # 准备标签
    query_labels = torch.from_numpy(query_data['labels']).float()
    database_labels = torch.from_numpy(database_data['labels']).float()

    if use_gpu:
        query_labels = query_labels.cuda()
        database_labels = database_labels.cuda()

    # ========== 明文 mAP 评估 ==========
    print(f"\n{'='*40}")
    print("明文 mAP 评估")
    print(f"{'='*40}")

    # 图像哈希码 mAP
    img_map = calc_map_k(img_codes_query, img_codes_db, query_labels, database_labels)
    print(f"图像哈希码 mAP: {img_map:.6f}")

    # 文本哈希码 mAP
    txt_map = calc_map_k(txt_codes_query, txt_codes_db, query_labels, database_labels)
    print(f"文本哈希码 mAP: {txt_map:.6f}")

    # ========== ASPE 加密 ==========
    print(f"\n{'='*40}")
    print("ASPE 加密")
    print(f"{'='*40}")

    aspe = ASPEForDCMH(bit_dim=bit, seed=42)

    # 加密图像哈希码
    print("加密图像哈希码...")
    enc_img_db = aspe.GenEnc(img_codes_db.numpy())
    enc_img_query = aspe.GenTrap(img_codes_query.numpy())

    # 加密文本哈希码
    print("加密文本哈希码...")
    enc_txt_db = aspe.GenEnc(txt_codes_db.numpy())
    enc_txt_query = aspe.GenTrap(txt_codes_query.numpy())

    # ========== 密文 mAP 评估 ==========
    print(f"\n{'='*40}")
    print("密文 mAP 评估")
    print(f"{'='*40}")

    # 图像密文 mAP
    enc_img_map = aspe.calc_ciphertext_map(
        enc_img_query, enc_img_db,
        query_labels.numpy(), database_labels.numpy()
    )
    print(f"加密图像 mAP: {enc_img_map:.6f}")

    # 文本密文 mAP
    enc_txt_map = aspe.calc_ciphertext_map(
        enc_txt_query, enc_txt_db,
        query_labels.numpy(), database_labels.numpy()
    )
    print(f"加密文本 mAP: {enc_txt_map:.6f}")

    # ========== 验证排序一致性 ==========
    print(f"\n{'='*40}")
    print("ASPE 排序一致性验证")
    print(f"{'='*40}")

    # 采样验证
    num_samples = min(10, len(img_codes_query))
    img_sort_consistent = aspe.verify_sorting_consistency(
        img_codes_query.numpy()[:num_samples],
        img_codes_db.numpy(),
        num_samples
    )
    print(f"图像排序一致性：{'✓ 通过' if img_sort_consistent else '✗ 失败'}")

    txt_sort_consistent = aspe.verify_sorting_consistency(
        txt_codes_query.numpy()[:num_samples],
        txt_codes_db.numpy(),
        num_samples
    )
    print(f"文本排序一致性：{'✓ 通过' if txt_sort_consistent else '✗ 失败'}")

    # ========== 生成报告 ==========
    print(f"\n{'='*60}")
    print("DCMH + ASPE 快速评估报告")
    print(f"{'='*60}")

    print(f"\n【配置】")
    print(f"  数据集：FLICKR-25K")
    print(f"  哈希码位数：{bit}")
    print(f"  查询集大小：{len(query_data['images'])}")
    print(f"  数据库大小：{len(database_data['images'])}")
    print(f"  预训练模型：{'imagenet-vgg-f.mat' if pretrain_data else '无（随机初始化）'}")

    print(f"\n【明文性能】")
    print(f"  图像 mAP: {img_map:.6f}")
    print(f"  文本 mAP: {txt_map:.6f}")

    print(f"\n【密文性能】")
    print(f"  加密图像 mAP: {enc_img_map:.6f}")
    print(f"  加密文本 mAP: {enc_txt_map:.6f}")

    print(f"\n【一致性验证】")
    img_diff = abs(img_map - enc_img_map)
    txt_diff = abs(txt_map - enc_txt_map)
    print(f"  图像 mAP 差异：{img_diff:.8f} {'✓' if img_diff < 1e-6 else '✗'}")
    print(f"  文本 mAP 差异：{txt_diff:.8f} {'✓' if txt_diff < 1e-6 else '✗'}")

    print(f"\n【结论】")
    if img_diff < 1e-6 and txt_diff < 1e-6 and img_sort_consistent and txt_sort_consistent:
        print("  ✓ ASPE 加密正确性验证通过！")
        print("  密文检索与明文检索结果完全一致。")
    else:
        print("  ✗ ASPE 加密验证失败，请检查实现。")

    print(f"\n{'='*60}\n")

    return {
        'plaintext_img_map': img_map,
        'plaintext_txt_map': txt_map,
        'ciphertext_img_map': enc_img_map,
        'ciphertext_txt_map': enc_txt_map,
        'img_map_diff': img_diff,
        'txt_map_diff': txt_diff,
        'img_sort_consistent': img_sort_consistent,
        'txt_sort_consistent': txt_sort_consistent
    }


def main():
    parser = argparse.ArgumentParser(description='DCMH + ASPE 快速评估')
    parser.add_argument('--mode', type=str, default='quick', choices=['quick', 'full'],
                        help='评估模式：quick（快速评估）或 full（完整训练后评估）')
    parser.add_argument('--bit', type=int, default=64, help='哈希码位数')
    parser.add_argument('--epochs', type=int, default=20, help='训练轮数（仅 full 模式）')
    parser.add_argument('--data-path', type=str, default='data/FLICKR-25K.mat',
                        help='FLICKR-25K 数据集路径')
    parser.add_argument('--pretrain-path', type=str, default='data/imagenet-vgg-f.mat',
                        help='预训练模型路径')

    args = parser.parse_args()

    if args.mode == 'quick':
        results = quick_eval(
            bit=args.bit,
            data_path=args.data_path,
            pretrain_path=args.pretrain_path
        )
    else:
        print("Full 模式暂未实现，请使用 quick 模式。")
        print("使用命令：python evaluation/quick_eval_dcmh_aspe.py --mode quick --bit 64")
        return

    # 保存结果
    import json
    output_file = f'evaluation/results_dcmh_aspe_bit{args.bit}.json'
    with open(output_file, 'w') as f:
        json.dump({k: float(v) if isinstance(v, (np.floating, float)) else bool(v)
                   for k, v in results.items()}, f, indent=2)
    print(f"结果已保存到：{output_file}")


if __name__ == '__main__':
    main()
