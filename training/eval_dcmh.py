"""
DCMH 评估脚本

训练完成后单独运行评估，计算 mAP 指标。
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
from datetime import datetime

from config.dcmh_config import DCMHConfig
from core.hashing.dcmh_image import build_dcmh_image_model
from core.hashing.dcmh_text import build_dcmh_text_model
from core.retrieval.dcmh_metrics import calc_map_k
from training.dcmh_dataset import DCMHImageDataset, DCMHTextDataset


class EvalConfig(DCMHConfig):
    """评估配置"""
    # 模型路径
    img_model_path = None  # 必须指定
    txt_model_path = None  # 必须指定

    # 数据参数
    batch_size = 128
    num_workers = 0


def evaluate(**kwargs):
    """
    评估训练好的模型。

    计算图像到文本（i2t）和文本到图像（t2i）的 mAP 指标。
    """
    opt = EvalConfig()
    opt.parse(kwargs)

    print("=" * 60)
    print("DCMH 模型评估")
    print("=" * 60)

    # 验证模型路径
    if opt.img_model_path is None or opt.txt_model_path is None:
        print("错误：必须指定 --img_model_path 和 --txt_model_path")
        return

    if not os.path.exists(opt.img_model_path):
        print(f"错误：图像模型文件不存在：{opt.img_model_path}")
        return

    if not os.path.exists(opt.txt_model_path):
        print(f"错误：文本模型文件不存在：{opt.txt_model_path}")
        return

    # 数据划分索引
    query_indices = np.arange(0, opt.query_size)
    train_indices = np.arange(opt.query_size, opt.query_size + opt.training_size)
    retrieval_indices = np.arange(opt.query_size, opt.query_size + opt.database_size)

    # 创建 Dataset
    print("\n创建数据集...")
    query_img_dataset = DCMHImageDataset(opt.data_path, query_indices, transform=None)
    retrieval_img_dataset = DCMHImageDataset(opt.data_path, retrieval_indices, transform=None)
    query_txt_dataset = DCMHTextDataset(opt.data_path, query_indices)
    retrieval_txt_dataset = DCMHTextDataset(opt.data_path, retrieval_indices)

    # 创建 DataLoader
    query_img_loader = DataLoader(query_img_dataset, batch_size=opt.batch_size,
                                   shuffle=False, num_workers=opt.num_workers)
    retrieval_img_loader = DataLoader(retrieval_img_dataset, batch_size=opt.batch_size,
                                       shuffle=False, num_workers=opt.num_workers)

    # 加载标签
    query_L = torch.from_numpy(query_txt_dataset.h5_file['LAll'][query_indices]).float()
    retrieval_L = torch.from_numpy(retrieval_txt_dataset.h5_file['LAll'][retrieval_indices]).float()

    # 获取 y_dim
    y_dim = query_txt_dataset.h5_file['YAll'].shape[1]

    print(f"  query_size: {len(query_indices)}")
    print(f"  retrieval_size: {len(retrieval_indices)}")
    print(f"  y_dim: {y_dim}")

    # 构建模型
    print("\n正在构建模型...")
    img_model = build_dcmh_image_model(bit=opt.bit, pretrain_model_path=None)
    txt_model = build_dcmh_text_model(y_dim=y_dim, bit=opt.bit)

    # 加载权重
    print(f"\n加载图像模型权重：{opt.img_model_path}")
    img_model.load_state_dict(torch.load(opt.img_model_path, map_location='cpu', weights_only=True))

    print(f"加载文本模型权重：{opt.txt_model_path}")
    txt_model.load_state_dict(torch.load(opt.txt_model_path, map_location='cpu', weights_only=True))

    if opt.use_gpu and torch.cuda.is_available():
        img_model = img_model.cuda()
        txt_model = txt_model.cuda()
        device = 'cuda'
        query_L = query_L.cuda()
        retrieval_L = retrieval_L.cuda()
    else:
        device = 'cpu'
    print(f"模型加载完成！设备：{device}")

    # 生成哈希码
    print("\n生成图像哈希码...")
    qBX = generate_image_code_from_loader(img_model, query_img_loader, opt.bit, opt.use_gpu)
    rBX = generate_image_code_from_loader(img_model, retrieval_img_loader, opt.bit, opt.use_gpu)

    print("生成文本哈希码...")
    qBY = generate_text_code_from_dataset(txt_model, query_txt_dataset, opt.bit, opt.use_gpu)
    rBY = generate_text_code_from_dataset(txt_model, retrieval_txt_dataset, opt.bit, opt.use_gpu)

    # 计算 mAP
    print("\n计算 mAP 指标...")
    mapi2t = calc_map_k(qBX, rBY, query_L, retrieval_L)
    mapt2i = calc_map_k(qBY, rBX, query_L, retrieval_L)

    print("\n" + "=" * 60)
    print("评估结果")
    print("=" * 60)
    print(f"MAP(i->t): {mapi2t:.4f}")
    print(f"MAP(t->i): {mapt2i:.4f}")

    # 保存结果
    result = {
        'img_model_path': opt.img_model_path,
        'txt_model_path': opt.txt_model_path,
        'map_i2t': float(mapi2t),
        'map_t2i': float(mapt2i),
        'eval_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    # 保存到模型同目录
    result_dir = os.path.dirname(opt.img_model_path)
    with open(os.path.join(result_dir, 'eval_result.json'), 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\n评估结果保存到：{os.path.join(result_dir, 'eval_result.json')}")

    return mapi2t, mapt2i


def generate_image_code_from_loader(img_model, loader, bit, use_gpu):
    """从 DataLoader 生成图像哈希码"""
    num_data = len(loader.dataset)
    B = torch.zeros(num_data, bit, dtype=torch.float)
    if use_gpu and torch.cuda.is_available():
        B = B.cuda()

    img_model.eval()
    with torch.no_grad():
        for batch_imgs, indices in tqdm(loader, desc='Generating image codes'):
            # 类型验证
            if not isinstance(indices, torch.Tensor):
                raise TypeError(f"indices must be torch.Tensor, got {type(indices).__name__}")

            # 范围验证
            if indices.min() < 0 or indices.max() >= num_data:
                raise IndexError(f"indices out of range: [{indices.min()}, {indices.max()}] vs [0, {num_data})")

            image = batch_imgs.type(torch.float)
            if use_gpu and torch.cuda.is_available():
                image = image.cuda()
            cur_f = img_model(image)
            B[indices, :] = cur_f.data

    B = torch.sign(B)
    return B


def generate_text_code_from_dataset(txt_model, dataset, bit, use_gpu):
    """从 Dataset 生成文本哈希码"""
    num_data = len(dataset)
    B = torch.zeros(num_data, bit, dtype=torch.float)
    if use_gpu and torch.cuda.is_available():
        B = B.cuda()

    txt_model.eval()
    with torch.no_grad():
        for i in tqdm(range(num_data), desc='Generating text codes'):
            tags, labels, idx = dataset[i]
            text = tags.unsqueeze(0).type(torch.float)
            if use_gpu and torch.cuda.is_available():
                text = text.cuda()
            cur_g = txt_model(text)
            B[idx, :] = cur_g.data

    B = torch.sign(B)
    return B


if __name__ == '__main__':
    import fire
    fire.Fire(evaluate)