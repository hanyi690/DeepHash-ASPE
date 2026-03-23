"""
DCMH 评估脚本

训练完成后单独运行评估，计算 mAP 指标。
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from tqdm import tqdm
import json
from datetime import datetime

from config.dcmh_config import DCMHConfig
from core.hashing.dcmh_data_loader import load_data, split_data
from core.hashing.dcmh_image import build_dcmh_image_model
from core.hashing.dcmh_text import build_dcmh_text_model
from core.retrieval.dcmh_metrics import calc_map_k


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

    # 加载数据
    print("\n从原始数据加载...")
    images, tags, labels = load_data(opt.data_path)
    X, Y, L = split_data(images, tags, labels,
                         query_size=opt.query_size,
                         training_size=opt.training_size,
                         database_size=opt.database_size)

    train_features = X['train']
    query_features = X['query']
    retrieval_features = X['retrieval']

    train_labels = L['train']
    query_labels = L['query']
    retrieval_labels = L['retrieval']

    train_tags = Y['train']
    query_tags = Y['query']
    retrieval_tags = Y['retrieval']

    print(f"  train_features: {train_features.shape}")
    print(f"  query_features: {query_features.shape}")
    print(f"  retrieval_features: {retrieval_features.shape}")

    y_dim = train_tags.shape[1]

    # 构建模型
    print("\n正在构建模型...")
    img_model = build_dcmh_image_model(bit=opt.bit, pretrain_model_path=None)
    txt_model = build_dcmh_text_model(y_dim=y_dim, bit=opt.bit)

    # 加载权重
    print(f"\n加载图像模型权重：{opt.img_model_path}")
    img_model.load_state_dict(torch.load(opt.img_model_path, map_location='cpu'))

    print(f"加载文本模型权重：{opt.txt_model_path}")
    txt_model.load_state_dict(torch.load(opt.txt_model_path, map_location='cpu'))

    if opt.use_gpu and torch.cuda.is_available():
        img_model = img_model.cuda()
        txt_model = txt_model.cuda()
        device = 'cuda'
    else:
        device = 'cpu'
    print(f"模型加载完成！设备：{device}")

    # 转换为 tensor
    train_L = torch.from_numpy(train_labels).float()
    query_L = torch.from_numpy(query_labels).float()
    retrieval_L = torch.from_numpy(retrieval_labels).float()

    query_x = torch.from_numpy(query_features).float()
    retrieval_x = torch.from_numpy(retrieval_features).float()
    query_y = torch.from_numpy(query_tags).float()
    retrieval_y = torch.from_numpy(retrieval_tags).float()

    if opt.use_gpu and torch.cuda.is_available():
        train_L = train_L.cuda()
        query_L = query_L.cuda()
        retrieval_L = retrieval_L.cuda()
        query_x = query_x.cuda()
        retrieval_x = retrieval_x.cuda()
        query_y = query_y.cuda()
        retrieval_y = retrieval_y.cuda()

    # 生成哈希码
    print("\n生成图像哈希码...")
    qBX = generate_image_code(img_model, query_x, opt.bit, opt.batch_size, opt.use_gpu)
    rBX = generate_image_code(img_model, retrieval_x, opt.bit, opt.batch_size, opt.use_gpu)

    print("生成文本哈希码...")
    qBY = generate_text_code(txt_model, query_y, opt.bit, opt.batch_size, opt.use_gpu)
    rBY = generate_text_code(txt_model, retrieval_y, opt.bit, opt.batch_size, opt.use_gpu)

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
        'map_i2t': mapi2t,
        'map_t2i': mapt2i,
        'eval_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    # 保存到模型同目录
    result_dir = os.path.dirname(opt.img_model_path)
    with open(os.path.join(result_dir, 'eval_result.json'), 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\n评估结果保存到：{os.path.join(result_dir, 'eval_result.json')}")

    return mapi2t, mapt2i


def generate_image_code(img_model, X, bit, batch_size, use_gpu):
    """生成图像哈希码"""
    num_data = X.shape[0]
    index = np.linspace(0, num_data - 1, num_data).astype(int)
    B = torch.zeros(num_data, bit, dtype=torch.float)
    if use_gpu and torch.cuda.is_available():
        B = B.cuda()

    img_model.eval()
    with torch.no_grad():
        for i in range(num_data // batch_size + 1):
            ind = index[i * batch_size: min((i + 1) * batch_size, num_data)]
            image = X[ind].type(torch.float)
            if use_gpu and torch.cuda.is_available():
                image = image.cuda()
            cur_f = img_model(image)
            B[ind, :] = cur_f.data

    B = torch.sign(B)
    return B


def generate_text_code(txt_model, Y, bit, batch_size, use_gpu):
    """生成文本哈希码"""
    num_data = Y.shape[0]
    index = np.linspace(0, num_data - 1, num_data).astype(int)
    B = torch.zeros(num_data, bit, dtype=torch.float)
    if use_gpu and torch.cuda.is_available():
        B = B.cuda()

    txt_model.eval()
    with torch.no_grad():
        for i in range(num_data // batch_size + 1):
            ind = index[i * batch_size: min((i + 1) * batch_size, num_data)]
            text = Y[ind].unsqueeze(1).unsqueeze(-1).type(torch.float)
            if use_gpu and torch.cuda.is_available():
                text = text.cuda()
            cur_g = txt_model(text)
            B[ind, :] = cur_g.data

    B = torch.sign(B)
    return B


if __name__ == '__main__':
    import fire
    fire.Fire(evaluate)
