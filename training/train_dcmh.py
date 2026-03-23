"""
DCMH 训练脚本（Flickr25K）

基于 reference/DCMH/main.py 实现原始 DCMH 训练流程，使用重构后的项目结构。

内存优化模式（--low_memory=True）：
- 使用 Dataset 按需加载图像，内存从 10GB+ 降至 < 100MB
- 支持跳过训练中验证（--valid=False），训练后再单独评估
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from torch import nn
import torch.nn.functional as torch_func
from torch.autograd import Variable
from torch.optim import SGD
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from datetime import datetime

from config.dcmh_config import DCMHConfig
from core.hashing.dcmh_data_loader import load_data, load_pretrain_model, split_data
from core.hashing.dcmh_image import build_dcmh_image_model
from core.hashing.dcmh_text import build_dcmh_text_model
from core.retrieval.dcmh_metrics import calc_map_k


class TrainConfig(DCMHConfig):
    """训练配置（扩展自 DCMHConfig）"""
    # 内存优化选项
    low_memory = False  # 启用低内存模式（使用 Dataset 按需加载）

    # 数据加载优化
    # Windows 上 num_workers > 0 可能导致共享内存错误，默认使用 0
    num_workers = 0  # 数据加载线程数（0=单进程，Windows 推荐）
    pin_memory = True  # 加速 GPU 传输
    prefetch_factor = 2  # 预取因子（num_workers=0 时忽略）

    # 断点恢复
    checkpoint_interval = 10  # 每 N 个 epoch 保存一次检查点
    resume_from = None  # 从指定检查点恢复（目录路径）


def train(**kwargs):
    """
    训练 DCMH 模型（reference/DCMH 的原始训练流程）。

    训练流程：
    1. 初始化 F_buffer 和 G_buffer（存储所有训练样本的哈希码）
    2. 计算全局相似性矩阵 Sim = calc_neighbor(train_L, train_L)
    3. B = sign(F_buffer + G_buffer)
    4. 每个 epoch:
       - 图像训练：随机采样 batch，计算 cur_f，更新 F_buffer
       - 文本训练：随机采样 batch，计算 cur_g，更新 G_buffer
       - 更新 B = sign(F_buffer + G_buffer)
       - 计算总损失
    5. 学习率线性衰减
    """
    opt = TrainConfig()
    opt.parse(kwargs)

    print("=" * 60)
    print("DCMH 训练")
    print("=" * 60)

    y_dim = None
    train_L = query_L = retrieval_L = None
    train_y = query_y = retrieval_y = None
    num_train = 0

    # ========== 数据加载模式 ==========
    if opt.low_memory:
        # 模式 2：低内存模式（使用 Dataset 按需加载）
        print("\n低内存模式：使用 Dataset 按需加载图像")
        from training.dcmh_dataset import DCMHImageDataset, DCMHTextDataset
        from torchvision import transforms

        # 图像变换（与 reference 一致：只用 ToTensor，不做额外归一化）
        # DCMH 模型内部 forward 会做 x - self.mean，所以直接传入 [0, 1] 范围的 tensor
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),  # 转为 [0, 1] tensor，模型内部会减 mean
        ])

        # 数据划分索引
        query_indices = np.arange(0, opt.query_size)
        train_indices = np.arange(opt.query_size, opt.query_size + opt.training_size)
        retrieval_indices = np.arange(opt.query_size, opt.query_size + opt.database_size)

        # 创建 Dataset
        train_img_dataset = DCMHImageDataset(opt.data_path, train_indices, transform)
        query_img_dataset = DCMHImageDataset(opt.data_path, query_indices, transform)
        retrieval_img_dataset = DCMHImageDataset(opt.data_path, retrieval_indices, transform)

        train_txt_dataset = DCMHTextDataset(opt.data_path, train_indices)
        query_txt_dataset = DCMHTextDataset(opt.data_path, query_indices)
        retrieval_txt_dataset = DCMHTextDataset(opt.data_path, retrieval_indices)

        # 创建 DataLoader
        train_img_loader = DataLoader(train_img_dataset, batch_size=opt.batch_size,
                                      shuffle=True, num_workers=opt.num_workers,
                                      pin_memory=opt.pin_memory)
        query_img_loader = DataLoader(query_img_dataset, batch_size=opt.batch_size,
                                      shuffle=False, num_workers=opt.num_workers,
                                      pin_memory=opt.pin_memory)
        retrieval_img_loader = DataLoader(retrieval_img_dataset, batch_size=opt.batch_size,
                                          shuffle=False, num_workers=opt.num_workers,
                                          pin_memory=opt.pin_memory)

        # 获取 y_dim
        y_dim = train_txt_dataset.h5_file['YAll'].shape[1]

        # 加载标签
        train_L = torch.from_numpy(train_txt_dataset.h5_file['LAll'][train_indices]).float()
        query_L = torch.from_numpy(query_txt_dataset.h5_file['LAll'][query_indices]).float()
        retrieval_L = torch.from_numpy(retrieval_txt_dataset.h5_file['LAll'][retrieval_indices]).float()

        train_y = torch.from_numpy(train_txt_dataset.h5_file['YAll'][train_indices]).float()
        query_y = torch.from_numpy(query_txt_dataset.h5_file['YAll'][query_indices]).float()
        retrieval_y = torch.from_numpy(retrieval_txt_dataset.h5_file['YAll'][retrieval_indices]).float()

        num_train = len(train_indices)

        print(f"  train_indices: {num_train}")
        print(f"  使用按需加载，内存占用 < 100MB！")

        # 将 train_x 设置为 None，训练时从 DataLoader 获取
        train_x = None
        query_x = None
        retrieval_x = None

    else:
        # 模式 3：标准模式（加载所有数据到内存）
        print("\n标准模式：加载所有数据到内存")
        # 加载数据
        images, tags, labels = load_data(opt.data_path)
        pretrain_model = load_pretrain_model(opt.pretrain_model_path)
        y_dim = tags.shape[1]

        X, Y, L = split_data(images, tags, labels,
                          query_size=opt.query_size,
                          training_size=opt.training_size,
                          database_size=opt.database_size)
        print('...loading and splitting data finish')

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

        num_train = train_x.shape[0]

    # ========== 构建模型 ==========
    print("\n正在构建模型...")
    img_model = build_dcmh_image_model(bit=opt.bit, pretrain_model_path=opt.pretrain_model_path)
    txt_model = build_dcmh_text_model(y_dim=y_dim, bit=opt.bit)

    # 加载已保存的模型（如果指定）
    if opt.load_img_path:
        img_model.load(opt.load_img_path, use_gpu=opt.use_gpu)
        print(f"已加载图像模型: {opt.load_img_path}")
    if opt.load_txt_path:
        txt_model.load(opt.load_txt_path, use_gpu=opt.use_gpu)
        print(f"已加载文本模型: {opt.load_txt_path}")

    if opt.use_gpu and torch.cuda.is_available():
        img_model = img_model.cuda()
        txt_model = txt_model.cuda()
        device = 'cuda'
    else:
        device = 'cpu'
    print(f"模型构建完成！设备：{device}")

    # 移动到 GPU
    if opt.use_gpu and torch.cuda.is_available():
        train_L = train_L.cuda()
        query_L = query_L.cuda()
        retrieval_L = retrieval_L.cuda()

        if train_x is not None:
            train_x = train_x.cuda()
            query_x = query_x.cuda()
            retrieval_x = retrieval_x.cuda()

        train_y = train_y.cuda()
        query_y = query_y.cuda()
        retrieval_y = retrieval_y.cuda()

    # ========== 初始化变量 ==========
    batch_size = opt.batch_size
    num_train = train_L.shape[0]
    lr = opt.lr
    max_mapi2t = max_mapt2i = 0.
    result_dir = None

    # 创建优化器
    optimizer_img = SGD(img_model.parameters(), lr=lr)
    optimizer_txt = SGD(txt_model.parameters(), lr=lr)

    # ========== 检查点恢复 ==========
    start_epoch = 0
    checkpoint_loaded = False

    if opt.resume_from:
        checkpoint_path = os.path.join(opt.resume_from, 'checkpoint.pth')
        if os.path.exists(checkpoint_path):
            print(f"\n从检查点恢复: {checkpoint_path}")
            checkpoint = load_checkpoint(checkpoint_path, img_model, txt_model,
                                        optimizer_img, optimizer_txt, device)
            start_epoch = checkpoint['epoch'] + 1
            max_mapi2t = checkpoint['max_mapi2t']
            max_mapt2i = checkpoint['max_mapt2i']
            lr = checkpoint['lr']
            result_dir = opt.resume_from
            checkpoint_loaded = True
            print(f"已从 epoch {start_epoch} 恢复训练")
            print(f"最佳 mAP: I2T={max_mapi2t:.4f}, T2I={max_mapt2i:.4f}")
        else:
            print(f"警告: 检查点文件不存在 {checkpoint_path}，从头开始训练")

    # ========== 初始化缓冲区 ==========
    # 计算全局相似性矩阵（始终需要）
    Sim = calc_neighbor(train_L, train_L)

    # 如果从检查点恢复，加载缓冲区
    if checkpoint_loaded:
        F_buffer = checkpoint['F_buffer'].to(device)
        G_buffer = checkpoint['G_buffer'].to(device)
        B = checkpoint['B'].to(device)
    else:
        F_buffer = torch.randn(num_train, opt.bit)
        G_buffer = torch.randn(num_train, opt.bit)

        if opt.use_gpu and torch.cuda.is_available():
            F_buffer = F_buffer.cuda()
            G_buffer = G_buffer.cuda()

        B = torch.sign(F_buffer + G_buffer)

    # 学习率线性衰减
    learning_rate = np.linspace(opt.lr, np.power(10, -6.), opt.max_epoch + 1)
    result = {
        'loss': []
    }

    ones = torch.ones(batch_size, 1)
    ones_ = torch.ones(num_train - batch_size, 1)

    # 创建结果目录（如果不从检查点恢复）
    if result_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        result_dir = os.path.join('results/flickr-25k', timestamp)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    print("\n" + "=" * 60)
    print("开始训练...")
    print("=" * 60)
    print(f"训练集大小：{num_train}")
    print(f"批次大小：{batch_size}")
    print(f"最大 Epoch: {opt.max_epoch}")
    print(f"低内存模式：{opt.low_memory}")
    print(f"训练中验证：{opt.valid}")
    print("=" * 60)

    for epoch in range(start_epoch, opt.max_epoch):
        # ========== 训练图像网络 ==========
        if opt.low_memory:
            # 低内存模式：使用 DataLoader
            img_model.train()
            loader_iter = iter(train_img_loader)

            for i in tqdm(range(num_train // batch_size), desc=f'Epoch {epoch+1}/{opt.max_epoch} [Img]'):
                try:
                    batch_imgs = next(loader_iter)
                except StopIteration:
                    loader_iter = iter(train_img_loader)
                    batch_imgs = next(loader_iter)

                # 获取索引
                if isinstance(batch_imgs, (list, tuple)):
                    batch_imgs, indices = batch_imgs
                else:
                    # 需要从 dataset 获取索引
                    pass

                image = Variable(batch_imgs.type(torch.float))
                if opt.use_gpu and torch.cuda.is_available():
                    image = image.cuda()

                # 获取对应的标签
                sample_L = Variable(train_L[indices, :])

                # 移动变量到 GPU（如果需要）
                if opt.use_gpu and torch.cuda.is_available():
                    sample_L = sample_L.cuda()
                    ones = ones.cuda()
                    ones_ = ones_.cuda()

                # 计算相似性矩阵
                S = calc_neighbor(sample_L, train_L)

                # 前向传播
                cur_f = img_model(image)

                # 更新 F_buffer
                F_buffer[indices, :] = cur_f.data
                F = Variable(F_buffer)
                G = Variable(G_buffer)

                # 计算损失
                unupdated_ind = np.setdiff1d(range(num_train), indices)
                theta_x = 1.0 / 2 * torch.matmul(cur_f, G.t())
                logloss_x = -torch.sum(S * theta_x - torch_func.softplus(theta_x))
                quantization_x = torch.sum(torch.pow(B[indices, :] - cur_f, 2))
                balance_x = torch.sum(torch.pow(cur_f.t().mm(ones) + F[unupdated_ind].t().mm(ones_), 2))
                loss_x = logloss_x + opt.gamma * quantization_x + opt.eta * balance_x
                loss_x /= (batch_size * num_train)

                optimizer_img.zero_grad()
                loss_x.backward()
                torch.nn.utils.clip_grad_norm_(img_model.parameters(), max_norm=5.0)
                optimizer_img.step()
        else:
            # 标准模式：使用内存中的数据
            img_model.train()
            for i in tqdm(range(num_train // batch_size), desc=f'Epoch {epoch+1}/{opt.max_epoch} [Img]'):
                index = np.random.permutation(num_train)
                ind = index[0: batch_size]
                unupdated_ind = np.setdiff1d(range(num_train), ind)

                sample_L = Variable(train_L[ind, :])
                image = Variable(train_x[ind].type(torch.float))
                if opt.use_gpu:
                    image = image.cuda()
                    sample_L = sample_L.cuda()
                    ones = ones.cuda()
                    ones_ = ones_.cuda()

                S = calc_neighbor(sample_L, train_L)
                cur_f = img_model(image)
                F_buffer[ind, :] = cur_f.data
                F = Variable(F_buffer)
                G = Variable(G_buffer)

                theta_x = 1.0 / 2 * torch.matmul(cur_f, G.t())
                logloss_x = -torch.sum(S * theta_x - torch_func.softplus(theta_x))
                quantization_x = torch.sum(torch.pow(B[ind, :] - cur_f, 2))
                balance_x = torch.sum(torch.pow(cur_f.t().mm(ones) + F[unupdated_ind].t().mm(ones_), 2))
                loss_x = logloss_x + opt.gamma * quantization_x + opt.eta * balance_x
                loss_x /= (batch_size * num_train)

                optimizer_img.zero_grad()
                loss_x.backward()
                torch.nn.utils.clip_grad_norm_(img_model.parameters(), max_norm=5.0)
                optimizer_img.step()

        # ========== 训练文本网络 ==========
        txt_model.train()
        for i in tqdm(range(num_train // batch_size), desc=f'Epoch {epoch+1}/{opt.max_epoch} [Txt]'):
            index = np.random.permutation(num_train)
            ind = index[0: batch_size]
            unupdated_ind = np.setdiff1d(range(num_train), ind)

            sample_L = Variable(train_L[ind, :])
            text = train_y[ind, :].unsqueeze(1).unsqueeze(-1).type(torch.float)
            text = Variable(text)
            if opt.use_gpu:
                text = text.cuda()
                sample_L = sample_L.cuda()

            S = calc_neighbor(sample_L, train_L)
            cur_g = txt_model(text)
            G_buffer[ind, :] = cur_g.data
            F = Variable(F_buffer)
            G = Variable(G_buffer)

            theta_y = 1.0 / 2 * torch.matmul(cur_g, F.t())
            logloss_y = -torch.sum(S * theta_y - torch_func.softplus(theta_y))
            quantization_y = torch.sum(torch.pow(B[ind, :] - cur_g, 2))
            balance_y = torch.sum(torch.pow(cur_g.t().mm(ones) + G[unupdated_ind].t().mm(ones_), 2))
            loss_y = logloss_y + opt.gamma * quantization_y + opt.eta * balance_y
            loss_y /= (num_train * batch_size)

            optimizer_txt.zero_grad()
            loss_y.backward()
            torch.nn.utils.clip_grad_norm_(txt_model.parameters(), max_norm=5.0)
            optimizer_txt.step()

        # ========== 更新 B ==========
        B = torch.sign(F_buffer + G_buffer)

        # ========== 计算总损失 ==========
        F = Variable(F_buffer)
        G = Variable(G_buffer)
        loss = calc_loss(B, F, G, Variable(Sim), opt.gamma, opt.eta)

        # NaN/Inf 检测
        if torch.isnan(loss) or torch.isinf(loss):
            print(f'Warning: loss is {loss.data}, skipping epoch {epoch + 1}')
            continue

        print('...epoch: %3d, loss: %3.3f, lr: %f' % (epoch + 1, loss.data, lr))
        result['loss'].append(float(loss.data))

        # ========== 验证 ==========
        if opt.valid:
            # 需要生成查询和检索集的哈希码
            if opt.low_memory:
                # 低内存模式：从 Dataset 生成
                qBX = generate_image_code_from_loader(img_model, query_img_loader, opt.bit, opt.use_gpu)
                qBY = generate_text_code_from_dataset(txt_model, query_txt_dataset, opt.bit, opt.use_gpu)
                rBX = generate_image_code_from_loader(img_model, retrieval_img_loader, opt.bit, opt.use_gpu)
                rBY = generate_text_code_from_dataset(txt_model, retrieval_txt_dataset, opt.bit, opt.use_gpu)
            else:
                qBX = generate_image_code(img_model, query_x, opt.bit, opt.batch_size, opt.use_gpu)
                qBY = generate_text_code(txt_model, query_y, opt.bit, opt.batch_size, opt.use_gpu)
                rBX = generate_image_code(img_model, retrieval_x, opt.bit, opt.batch_size, opt.use_gpu)
                rBY = generate_text_code(txt_model, retrieval_y, opt.bit, opt.batch_size, opt.use_gpu)

            mapi2t = calc_map_k(qBX, rBY, query_L, retrieval_L)
            mapt2i = calc_map_k(qBY, rBX, query_L, retrieval_L)

            print('...epoch: %3d, valid MAP: MAP(i->t): %3.4f, MAP(t->i): %3.4f' %
                  (epoch + 1, mapi2t, mapt2i))

            if mapt2i >= max_mapt2i and mapi2t >= max_mapi2t:
                max_mapi2t = mapi2t
                max_mapt2i = mapt2i
                save_model(img_model, 'img_model.pth', result_dir=result_dir, use_gpu=opt.use_gpu)
                save_model(txt_model, 'txt_model.pth', result_dir=result_dir, use_gpu=opt.use_gpu)

        # ========== 学习率衰减 ==========
        lr = learning_rate[epoch + 1]
        for param in optimizer_img.param_groups:
            param['lr'] = lr
        for param in optimizer_txt.param_groups:
            param['lr'] = lr

        # ========== 保存检查点 ==========
        if (epoch + 1) % opt.checkpoint_interval == 0:
            save_checkpoint(result_dir, epoch, img_model, txt_model,
                           optimizer_img, optimizer_txt,
                           F_buffer, G_buffer, B, lr,
                           max_mapi2t, max_mapt2i, use_gpu=opt.use_gpu)

    # ========== 训练完成 ==========
    print("\n" + "=" * 60)
    print("训练完成！")
    print("=" * 60)

    # 如果跳过了训练中验证，现在执行一次最终验证
    if not opt.valid:
        print("执行最终验证...")
        if opt.low_memory:
            qBX = generate_image_code_from_loader(img_model, query_img_loader, opt.bit, opt.use_gpu)
            qBY = generate_text_code_from_dataset(txt_model, query_txt_dataset, opt.bit, opt.use_gpu)
            rBX = generate_image_code_from_loader(img_model, retrieval_img_loader, opt.bit, opt.use_gpu)
            rBY = generate_text_code_from_dataset(txt_model, retrieval_txt_dataset, opt.bit, opt.use_gpu)
        else:
            qBX = generate_image_code(img_model, query_x, opt.bit, opt.batch_size, opt.use_gpu)
            qBY = generate_text_code(txt_model, query_y, opt.bit, opt.batch_size, opt.use_gpu)
            rBX = generate_image_code(img_model, retrieval_x, opt.bit, opt.batch_size, opt.use_gpu)
            rBY = generate_text_code(txt_model, retrieval_y, opt.bit, opt.batch_size, opt.use_gpu)

        mapi2t = calc_map_k(qBX, rBY, query_L, retrieval_L)
        mapt2i = calc_map_k(qBY, rBX, query_L, retrieval_L)

        print('   Final MAP: MAP(i->t): %3.4f, MAP(t->i): %3.4f' % (mapi2t, mapt2i))
        max_mapi2t = mapi2t
        max_mapt2i = mapt2i
        save_model(img_model, 'img_model.pth', result_dir=result_dir, use_gpu=opt.use_gpu)
        save_model(txt_model, 'txt_model.pth', result_dir=result_dir, use_gpu=opt.use_gpu)

    print('   max MAP: MAP(i->t): %3.4f, MAP(t->i): %3.4f' % (max_mapi2t, max_mapt2i))
    result['mapi2t'] = max_mapi2t
    result['mapt2i'] = max_mapt2i

    write_result(result, result_dir)


def valid(img_model, txt_model, query_x, retrieval_x, query_y, retrieval_y,
          query_L, retrieval_L, bit=64, batch_size=128, use_gpu=True):
    """验证函数：计算 mAP"""
    qBX = generate_image_code(img_model, query_x, bit, batch_size, use_gpu)
    qBY = generate_text_code(txt_model, query_y, bit, batch_size, use_gpu)
    rBX = generate_image_code(img_model, retrieval_x, bit, batch_size, use_gpu)
    rBY = generate_text_code(txt_model, retrieval_y, bit, batch_size, use_gpu)

    mapi2t = calc_map_k(qBX, rBY, query_L, retrieval_L)
    mapt2i = calc_map_k(qBY, rBX, query_L, retrieval_L)
    return mapi2t, mapt2i


def calc_neighbor(label1, label2):
    """
    计算相似性矩阵。

    如果两个样本有相同的标签，则相似度为 1，否则为 0。
    """
    if label1.is_cuda:
        Sim = (label1.matmul(label2.transpose(0, 1)) > 0).type(torch.cuda.FloatTensor)
    else:
        Sim = (label1.matmul(label2.transpose(0, 1)) > 0).type(torch.FloatTensor)
    return Sim


def calc_loss(B, F, G, Sim, gamma, eta):
    """
    计算 DCMH 总损失（归一化版本）。

    归一化方式与 batch loss 一致：除以 (num_train * num_train)

    包含：
    1. 对数损失（基于相似性矩阵）
    2. 量化损失
    3. 平衡损失
    """
    num_train = B.shape[0]
    theta = torch.matmul(F, G.transpose(0, 1)) / 2
    # 使用 softplus 替代 log(1+exp) 以增加数值稳定性
    term1 = torch.sum(torch_func.softplus(theta) - Sim * theta)
    term2 = torch.sum(torch.pow(B - F, 2) + torch.pow(B - G, 2))
    term3 = torch.sum(torch.pow(F.sum(dim=0), 2) + torch.pow(G.sum(dim=0), 2))

    # 归一化，防止数值溢出
    loss = (term1 + gamma * term2 + eta * term3) / (num_train * num_train)
    return loss


def generate_image_code(img_model, X, bit, batch_size=128, use_gpu=True):
    """生成图像哈希码"""
    num_data = X.shape[0]
    index = np.linspace(0, num_data - 1, num_data).astype(int)
    B = torch.zeros(num_data, bit, dtype=torch.float)
    if use_gpu and torch.cuda.is_available():
        B = B.cuda()

    img_model.eval()
    with torch.no_grad():
        for i in tqdm(range(num_data // batch_size + 1), desc='Generating image codes'):
            ind = index[i * batch_size: min((i + 1) * batch_size, num_data)]
            image = X[ind].type(torch.float)
            if use_gpu and torch.cuda.is_available():
                image = image.cuda()
            cur_f = img_model(image)
            B[ind, :] = cur_f.data

    B = torch.sign(B)
    return B


def generate_text_code(txt_model, Y, bit, batch_size=128, use_gpu=True):
    """生成文本哈希码"""
    num_data = Y.shape[0]
    index = np.linspace(0, num_data - 1, num_data).astype(int)
    B = torch.zeros(num_data, bit, dtype=torch.float)
    if use_gpu and torch.cuda.is_available():
        B = B.cuda()

    txt_model.eval()
    with torch.no_grad():
        for i in tqdm(range(num_data // batch_size + 1), desc='Generating text codes'):
            ind = index[i * batch_size: min((i + 1) * batch_size, num_data)]
            text = Y[ind].unsqueeze(1).unsqueeze(-1).type(torch.float)
            if use_gpu and torch.cuda.is_available():
                text = text.cuda()
            cur_g = txt_model(text)
            B[ind, :] = cur_g.data

    B = torch.sign(B)
    return B


def generate_image_code_from_loader(img_model, loader, bit, use_gpu):
    """从 DataLoader 生成图像哈希码（低内存模式）"""
    num_data = len(loader.dataset)
    B = torch.zeros(num_data, bit, dtype=torch.float)
    if use_gpu and torch.cuda.is_available():
        B = B.cuda()

    img_model.eval()
    with torch.no_grad():
        for batch_imgs, indices in tqdm(loader, desc='Generating image codes'):
            image = batch_imgs.type(torch.float)
            if use_gpu and torch.cuda.is_available():
                image = image.cuda()
            cur_f = img_model(image)
            B[indices, :] = cur_f.data

    B = torch.sign(B)
    return B


def generate_text_code_from_dataset(txt_model, dataset, bit, use_gpu):
    """从 Dataset 生成文本哈希码（低内存模式）"""
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


def save_checkpoint(result_dir, epoch, img_model, txt_model,
                    optimizer_img, optimizer_txt,
                    F_buffer, G_buffer, B, lr,
                    max_mapi2t, max_mapt2i, use_gpu=True):
    """
    保存训练检查点。

    参数：
        result_dir: 结果目录
        epoch: 当前 epoch
        img_model: 图像模型
        txt_model: 文本模型
        optimizer_img: 图像优化器
        optimizer_txt: 文本优化器
        F_buffer: 图像哈希缓冲区
        G_buffer: 文本哈希缓冲区
        B: 综合哈希码
        lr: 当前学习率
        max_mapi2t: 最佳 I2T mAP
        max_mapt2i: 最佳 T2I mAP
        use_gpu: 是否使用 GPU
    """
    checkpoint = {
        'epoch': epoch,
        'img_model_state': img_model.state_dict(),
        'txt_model_state': txt_model.state_dict(),
        'optimizer_img_state': optimizer_img.state_dict(),
        'optimizer_txt_state': optimizer_txt.state_dict(),
        'F_buffer': F_buffer.cpu() if use_gpu and torch.cuda.is_available() else F_buffer,
        'G_buffer': G_buffer.cpu() if use_gpu and torch.cuda.is_available() else G_buffer,
        'B': B.cpu() if use_gpu and torch.cuda.is_available() else B,
        'lr': lr,
        'max_mapi2t': max_mapi2t,
        'max_mapt2i': max_mapt2i,
    }
    path = os.path.join(result_dir, 'checkpoint.pth')
    torch.save(checkpoint, path)
    print(f'检查点已保存到 {path} (epoch {epoch + 1})')


def load_checkpoint(checkpoint_path, img_model, txt_model,
                    optimizer_img, optimizer_txt, device):
    """
    加载训练检查点。

    参数：
        checkpoint_path: 检查点文件路径
        img_model: 图像模型
        txt_model: 文本模型
        optimizer_img: 图像优化器
        optimizer_txt: 文本优化器
        device: 设备

    返回：
        checkpoint: 包含所有状态的字典
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    img_model.load_state_dict(checkpoint['img_model_state'])
    txt_model.load_state_dict(checkpoint['txt_model_state'])
    optimizer_img.load_state_dict(checkpoint['optimizer_img_state'])
    optimizer_txt.load_state_dict(checkpoint['optimizer_txt_state'])
    return checkpoint


def save_model(model, filename, result_dir=None, use_gpu=True):
    """保存模型"""
    if result_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        result_dir = os.path.join('results/flickr-25k', timestamp)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    filepath = os.path.join(result_dir, filename)
    if use_gpu and torch.cuda.is_available():
        torch.save(model.cpu().state_dict(), filepath)
        model.cuda()
    else:
        torch.save(model.state_dict(), filepath)
    print(f'Model saved to {filepath}')
    return result_dir


def write_result(result, result_dir=None):
    """保存结果"""
    if result_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        result_dir = os.path.join('results/flickr-25k', timestamp)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    # 转换 Tensor 为 Python 标量
    result_serializable = {}
    for k, v in result.items():
        if isinstance(v, torch.Tensor):
            result_serializable[k] = v.item()
        elif isinstance(v, list):
            result_serializable[k] = [x.item() if isinstance(x, torch.Tensor) else x for x in v]
        else:
            result_serializable[k] = v

    # 保存结果为 JSON 格式
    import json
    with open(os.path.join(result_dir, 'result.json'), 'w') as f:
        json.dump(result_serializable, f, indent=2)

    # 同时保存为文本格式（兼容旧格式）
    with open(os.path.join(result_dir, 'result.txt'), 'w') as f:
        for k, v in result.items():
            f.write(k + ' ' + str(v) + '\n')

    print(f'Result saved to {result_dir}')
    return result_dir


def help():
    """
    打印帮助的信息
    """
    print('''
    usage : python train_dcmh.py <function> [--args=value]
    <function> := train | test | help
    example:
            python train_dcmh.py train --lr=0.01
            python train_dcmh.py train --low_memory=True --valid=False
            python train_dcmh.py train --resume_from=results/flickr-25k/20260323_120612
            python train_dcmh.py help
    available args:''')
    for k, v in TrainConfig.__dict__.items():
        if not k.startswith('__') and not callable(v):
            print('\t\t{0}: {1}'.format(k, v))


if __name__ == '__main__':
    import fire
    fire.Fire()
