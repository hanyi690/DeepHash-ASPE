"""
DCMH 小批量训练测试脚本

用于验证 DCMH 实现的正确性，使用小规模数据集快速测试。

测试配置：
- training_size = 100 (原 10000)
- query_size = 50 (原 2000)
- database_size = 200 (原 18015)
- batch_size = 16 (原 128)
- max_epoch = 5 (原 500)
- bit = 16 (原 64)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch.optim import SGD
from tqdm import tqdm
from datetime import datetime
import json

from config.dcmh_config import DCMHConfig
from core.hashing.dcmh_data_loader import load_data, load_pretrain_model, split_data
from core.hashing.dcmh_image import build_dcmh_image_model
from core.hashing.dcmh_text import build_dcmh_text_model
from core.retrieval.dcmh_metrics import calc_map_k


class TestConfig(DCMHConfig):
    """小批量训练测试配置"""

    # 数据参数（缩小规模）
    training_size = 100  # 原 10000
    query_size = 50  # 原 2000
    database_size = 200  # 原 18015
    batch_size = 16  # 原 128

    # 超参数（减少 epoch 和 bit）
    max_epoch = 5  # 原 500
    bit = 16  # 原 64

    # 学习率
    lr = 10 ** (-1.5)

    # 验证
    valid = True

    # 保存结果
    save_result = True


def train_test(**kwargs):
    """
    小批量训练测试。

    验证 DCMH 实现的正确性：
    1. 损失应该下降
    2. mAP 应该逐渐提高
    3. 模型应该能够正常保存和加载
    """
    opt = TestConfig()
    opt.parse(kwargs)

    print("=" * 60)
    print("DCMH 小批量训练测试")
    print("=" * 60)

    # 尝试加载数据
    try:
        print("\n正在加载数据...")
        images, tags, labels = load_data(opt.data_path)
        print(f"数据加载成功！")
        print(f"  images: {images.shape}")
        print(f"  tags: {tags.shape}")
        print(f"  labels: {labels.shape}")
    except FileNotFoundError:
        print(f"\n警告：数据文件 {opt.data_path} 未找到")
        print("将使用虚拟数据进行测试...")

        # 生成虚拟数据
        total_size = opt.query_size + opt.training_size + opt.database_size
        images = np.random.randn(total_size, 3, 224, 224).astype(np.float32)
        tags = np.random.randint(0, 2, (total_size, 1000)).astype(np.float32)
        labels = np.random.randint(0, 2, (total_size, 10)).astype(np.float32)
        print(f"虚拟数据生成完成！")
        print(f"  images: {images.shape}")
        print(f"  tags: {tags.shape}")
        print(f"  labels: {labels.shape}")

    y_dim = tags.shape[1]

    # 数据划分
    X, Y, L = split_data(images, tags, labels,
                         query_size=opt.query_size,
                         training_size=opt.training_size,
                         database_size=opt.database_size)

    print(f"\n数据划分完成：")
    print(f"  query: {X['query'].shape}")
    print(f"  train: {X['train'].shape}")
    print(f"  database: {X['retrieval'].shape}")

    # 构建模型
    print("\n正在构建模型...")
    img_model = build_dcmh_image_model(bit=opt.bit, pretrain_model_path=None)
    txt_model = build_dcmh_text_model(y_dim=y_dim, bit=opt.bit)

    if opt.use_gpu and torch.cuda.is_available():
        img_model = img_model.cuda()
        txt_model = txt_model.cuda()
        device = 'cuda'
    else:
        device = 'cpu'
    print(f"模型构建完成！设备：{device}")

    # 准备训练数据
    train_L = torch.from_numpy(L['train'])
    train_x = torch.from_numpy(X['train'])
    train_y = torch.from_numpy(Y['train'])

    query_L = torch.from_numpy(L['query'])
    query_x = torch.from_numpy(X['query'])
    query_y = torch.from_numpy(Y['query'])

    retrieval_L = torch.from_numpy(L['retrieval'])
    retrieval_x = torch.from_numpy(X['retrieval'])
    retrieval_y = torch.from_numpy(Y['retrieval'])

    if opt.use_gpu and torch.cuda.is_available():
        train_L = train_L.cuda()
        query_L = query_L.cuda()
        retrieval_L = retrieval_L.cuda()

    num_train = train_x.shape[0]
    print(f"\n训练集大小：{num_train}")

    # 初始化缓冲区
    F_buffer = torch.randn(num_train, opt.bit)
    G_buffer = torch.randn(num_train, opt.bit)

    if opt.use_gpu and torch.cuda.is_available():
        F_buffer = F_buffer.cuda()
        G_buffer = G_buffer.cuda()

    # 计算全局相似性矩阵
    Sim = calc_neighbor(train_L, train_L)
    B = torch.sign(F_buffer + G_buffer)

    batch_size = opt.batch_size
    lr = opt.lr
    optimizer_img = SGD(img_model.parameters(), lr=lr)
    optimizer_txt = SGD(txt_model.parameters(), lr=lr)

    # 学习率线性衰减
    learning_rate = np.linspace(opt.lr, np.power(10, -6.), opt.max_epoch + 1)
    result = {
        'loss': [],
        'map_i2t': [],
        'map_t2i': []
    }

    ones = torch.ones(batch_size, 1)
    max_mapi2t = max_mapt2i = 0.

    print("\n" + "=" * 60)
    print("开始训练...")
    print("=" * 60)

    for epoch in range(opt.max_epoch):
        epoch_loss = 0.0
        num_batches = max(1, num_train // batch_size)

        # ========== 训练图像网络 ==========
        img_model.train()
        for i in range(num_batches):
            index = np.random.permutation(num_train)
            ind = index[0: min(batch_size, num_train)]
            unupdated_ind = np.setdiff1d(range(num_train), ind)
            actual_batch_size = len(ind)

            sample_L = Variable(train_L[ind, :])
            image = Variable(train_x[ind].type(torch.float))
            if opt.use_gpu and torch.cuda.is_available():
                image = image.cuda()
                sample_L = sample_L.cuda()

            # 计算相似性矩阵
            S = calc_neighbor(sample_L, train_L)

            # 前向传播
            cur_f = img_model(image)

            # 更新 F_buffer
            F_buffer[ind, :] = cur_f.data

            # 计算损失
            F = Variable(F_buffer)
            G = Variable(G_buffer)
            B_current = torch.sign(F_buffer + G_buffer)

            loss = calc_batch_loss(
                cur_f=cur_f,
                sample_S=S,
                G_buffer=G,
                F_buffer=F,
                B_buffer=B_current,
                ind=ind,
                unupdated_ind=unupdated_ind,
                batch_size=actual_batch_size,
                num_train=num_train,
                gamma=opt.gamma,
                eta=opt.eta
            )

            optimizer_img.zero_grad()
            loss.backward()
            optimizer_img.step()

            epoch_loss += loss.item()

        # ========== 训练文本网络 ==========
        txt_model.train()
        for i in range(num_batches):
            index = np.random.permutation(num_train)
            ind = index[0: min(batch_size, num_train)]
            unupdated_ind = np.setdiff1d(range(num_train), ind)
            actual_batch_size = len(ind)

            sample_L = Variable(train_L[ind, :])
            text = train_y[ind, :].unsqueeze(1).unsqueeze(-1).type(torch.float)
            text = Variable(text)
            if opt.use_gpu and torch.cuda.is_available():
                text = text.cuda()
                sample_L = sample_L.cuda()

            # 计算相似性矩阵
            S = calc_neighbor(sample_L, train_L)

            # 前向传播
            cur_g = txt_model(text)

            # 更新 G_buffer
            G_buffer[ind, :] = cur_g.data

            # 计算损失
            F = Variable(F_buffer)
            G = Variable(G_buffer)
            B_current = torch.sign(F_buffer + G_buffer)

            loss = calc_batch_loss(
                cur_f=cur_g,
                sample_S=S,
                G_buffer=F,  # 注意：这里交换 F 和 G
                F_buffer=G,
                B_buffer=B_current,
                ind=ind,
                unupdated_ind=unupdated_ind,
                batch_size=actual_batch_size,
                num_train=num_train,
                gamma=opt.gamma,
                eta=opt.eta
            )


            optimizer_txt.zero_grad()
            loss.backward()
            optimizer_txt.step()

            epoch_loss += loss.item()

        # ========== 更新 B ==========
        B = torch.sign(F_buffer + G_buffer)

        # ========== 计算平均损失 ==========
        avg_loss = epoch_loss / (2 * num_batches)
        result['loss'].append(avg_loss)

        print(f'\nEpoch {epoch + 1}/{opt.max_epoch}:')
        print(f'  Loss: {avg_loss:.4f}, LR: {lr:.6f}')

        # ========== 验证 ==========
        if opt.valid:
            img_model.eval()
            txt_model.eval()
            mapi2t, mapt2i = valid(
                img_model, txt_model,
                query_x, retrieval_x, query_y, retrieval_y,
                query_L, retrieval_L,
                opt.batch_size, opt.bit, opt.use_gpu
            )
            result['map_i2t'].append(mapi2t)
            result['map_t2i'].append(mapt2i)

            print(f'  MAP(i->t): {mapi2t:.4f}, MAP(t->i): {mapt2i:.4f}')

            if mapt2i >= max_mapt2i and mapi2t >= max_mapi2t:
                max_mapi2t = mapi2t
                max_mapt2i = mapt2i

        # ========== 学习率衰减 ==========
        lr = learning_rate[epoch + 1]
        for param in optimizer_img.param_groups:
            param['lr'] = lr
        for param in optimizer_txt.param_groups:
            param['lr'] = lr

    # ========== 训练完成 ==========
    print("\n" + "=" * 60)
    print("训练完成！")
    print("=" * 60)
    print(f'最佳 MAP: MAP(i->t): {max_mapi2t:.4f}, MAP(t->i): {max_mapt2i:.4f}')

    # 打印损失变化
    print("\n损失变化：")
    for i, loss in enumerate(result['loss']):
        print(f'  Epoch {i + 1}: {loss:.4f}')

    # 验证损失是否下降
    if len(result['loss']) >= 2:
        if result['loss'][-1] < result['loss'][0]:
            print("\n✓ 损失呈下降趋势，训练正常！")
        else:
            print("\n⚠ 损失未明显下降，可能需要调整超参数")

    # 保存结果
    if opt.save_result:
        # 将 tensor 转换为 float 以便 JSON 序列化
        result_json = {
            'loss': [float(x) for x in result['loss']],
            'map_i2t': [float(x) if torch.is_tensor(x) else x for x in result['map_i2t']],
            'map_t2i': [float(x) if torch.is_tensor(x) else x for x in result['map_t2i']],
            'max_mapi2t': float(max_mapi2t),
            'max_mapt2i': float(max_mapt2i)
        }
        save_result(result_json)

    return result


def valid(img_model, txt_model, query_x, retrieval_x, query_y, retrieval_y,
          query_L, retrieval_L, batch_size, bit, use_gpu):
    """验证函数：计算 mAP"""
    qBX = generate_image_code(img_model, query_x, bit, batch_size, use_gpu)
    qBY = generate_text_code(txt_model, query_y, bit, batch_size, use_gpu)
    rBX = generate_image_code(img_model, retrieval_x, bit, batch_size, use_gpu)
    rBY = generate_text_code(txt_model, retrieval_y, bit, batch_size, use_gpu)

    mapi2t = calc_map_k(qBX, rBY, query_L, retrieval_L)
    mapt2i = calc_map_k(qBY, rBX, query_L, retrieval_L)
    return mapi2t, mapt2i


def calc_neighbor(label1, label2):
    """计算相似性矩阵"""
    Sim = (label1.matmul(label2.transpose(0, 1)) > 0).type(torch.float32)
    return Sim


def calc_batch_loss(cur_f, sample_S, G_buffer, F_buffer, B_buffer,
                    ind, unupdated_ind, batch_size, num_train, gamma, eta):
    """
    计算小批量损失。

    包含：
    1. 对数损失
    2. 量化损失
    3. 平衡损失
    """
    # theta
    theta = 1.0 / 2 * torch.matmul(cur_f, G_buffer.t())

    # 对数损失
    log_loss = -torch.sum(sample_S * theta - torch.log(1.0 + torch.exp(theta)))

    # 量化损失
    quant_loss = torch.sum(torch.pow(B_buffer[ind, :] - cur_f, 2))

    # 平衡损失
    ones = torch.ones(batch_size, 1, device=cur_f.device)
    ones_ = torch.ones(num_train - batch_size, 1, device=cur_f.device)
    balance_loss = torch.sum(torch.pow(cur_f.t().mm(ones) + F_buffer[unupdated_ind].t().mm(ones_), 2))

    # 总损失
    loss = log_loss + gamma * quant_loss + eta * balance_loss
    loss = loss / (batch_size * num_train)

    return loss


def save_result(result, result_dir=None):
    """保存结果到 results/flickr-25k/时间戳 目录"""
    if result_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        result_dir = os.path.join('results/flickr-25k', timestamp)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    # 保存结果为 JSON 格式
    with open(os.path.join(result_dir, 'result.json'), 'w') as f:
        json.dump(result, f, indent=2)

    # 同时保存为文本格式
    with open(os.path.join(result_dir, 'result.txt'), 'w') as f:
        for k, v in result.items():
            f.write(k + ' ' + str(v) + '\n')

    print(f'Result saved to {result_dir}')
    return result_dir


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
    fire.Fire(train_test)
