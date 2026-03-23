"""
DCMH 训练脚本（Flickr25K）

基于 reference/DCMH/main.py 实现原始 DCMH 训练流程，使用重构后的项目结构。

使用 Dataset 按需加载图像，内存占用 < 100MB。
支持跳过训练中验证（--valid=False），训练后再单独评估。
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
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import os
from datetime import datetime

from config.dcmh_config import DCMHConfig
from core.hashing.dcmh_data_loader import load_pretrain_model
from core.hashing.dcmh_image import build_dcmh_image_model
from core.hashing.dcmh_text import build_dcmh_text_model
from core.retrieval.dcmh_metrics import calc_map_k


class TrainConfig(DCMHConfig):
    """训练配置（扩展自 DCMHConfig）"""

    # 数据加载优化
    # Windows 上 num_workers > 0 可能导致共享内存错误，默认使用 0
    num_workers = 0  # 数据加载线程数（0=单进程，Windows 推荐）
    pin_memory = True  # 加速 GPU 传输
    prefetch_factor = 2  # 预取因子（num_workers=0 时忽略）

    # 断点恢复
    checkpoint_interval = 10  # 每 N 个 epoch 保存一次检查点
    resume_from = None  # 从指定检查点恢复（目录路径）

    # 训练优化（PyTorch 2.0+）
    use_amp = True  # 混合精度训练（自动检测 GPU）
    use_compile = True  # torch.compile 优化（PyTorch 2.0+）


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

    # ========== 数据加载 ==========
    print("\n使用 Dataset 按需加载图像")
    from training.dcmh_dataset import DCMHImageDataset, DCMHTextDataset

    # 数据划分索引
    query_indices = np.arange(0, opt.query_size)
    train_indices = np.arange(opt.query_size, opt.query_size + opt.training_size)
    retrieval_indices = np.arange(opt.query_size, opt.query_size + opt.database_size)

    # 创建 Dataset（图像已经是 tensor 格式，不需要额外 transform）
    train_img_dataset = DCMHImageDataset(opt.data_path, train_indices)
    query_img_dataset = DCMHImageDataset(opt.data_path, query_indices)
    retrieval_img_dataset = DCMHImageDataset(opt.data_path, retrieval_indices)

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

        # torch.compile 优化（PyTorch 2.0+，仅 Linux 支持）
        # Windows 上 Triton 不可用，跳过 compile
        if opt.use_compile and hasattr(torch, 'compile'):
            import platform
            if platform.system() != 'Windows':
                try:
                    print("正在编译模型 (torch.compile)...")
                    img_model = torch.compile(img_model, mode='reduce-overhead')
                    txt_model = torch.compile(txt_model, mode='reduce-overhead')
                    print("模型编译完成！")
                except Exception as e:
                    print(f"torch.compile 失败，使用原始模型: {e}")
            else:
                print("Windows 平台不支持 torch.compile，跳过编译")
    else:
        device = 'cpu'
    print(f"模型构建完成！设备：{device}")

    # 移动到 GPU
    if opt.use_gpu and torch.cuda.is_available():
        train_L = train_L.cuda()
        query_L = query_L.cuda()
        retrieval_L = retrieval_L.cuda()
        train_y = train_y.cuda()
        query_y = query_y.cuda()
        retrieval_y = retrieval_y.cuda()

    # ========== 初始化变量 ==========
    batch_size = opt.batch_size
    num_train = train_L.shape[0]
    lr = opt.lr
    max_mapi2t = max_mapt2i = 0.
    result_dir = None

    # 创建优化器（添加 momentum 和 weight_decay，匹配 Matlab 原始实现）
    optimizer_img = SGD(img_model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    optimizer_txt = SGD(txt_model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

    # 混合精度训练 GradScaler
    use_amp = opt.use_amp and opt.use_gpu and torch.cuda.is_available()
    scaler = GradScaler('cuda', enabled=use_amp) if use_amp else None
    if use_amp:
        print("已启用混合精度训练 (AMP)")

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
        F_buffer = torch.zeros(num_train, opt.bit)
        G_buffer = torch.zeros(num_train, opt.bit)

        if opt.use_gpu and torch.cuda.is_available():
            F_buffer = F_buffer.cuda()
            G_buffer = G_buffer.cuda()

        B = torch.sign(F_buffer + G_buffer)

    # 学习率线性衰减
    learning_rate = np.linspace(opt.lr, np.power(10, -6.), opt.max_epoch + 1)
    result = {
        'loss': [],
        'mAP_history': []  # 新增：记录每次验证的 mAP
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
    print(f"验证间隔：每 {opt.valid_interval} 个 epoch" if opt.valid_interval > 0 else "训练中验证：禁用")
    print("=" * 60)

    for epoch in range(start_epoch, opt.max_epoch):
        # ========== 训练图像网络 ==========
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

            # 混合精度训练
            with autocast('cuda', enabled=use_amp):
                # 前向传播
                cur_f = img_model(image)

                # 计算损失
                unupdated_ind = np.setdiff1d(range(num_train), indices)
                theta_x = 1.0 / 2 * torch.matmul(cur_f, G_buffer.t())
                logloss_x = -torch.sum(S * theta_x - torch_func.softplus(theta_x))
                quantization_x = torch.sum(torch.pow(B[indices, :] - cur_f, 2))
                balance_x = torch.sum(torch.pow(cur_f.t().mm(ones) + F_buffer[unupdated_ind].t().mm(ones_), 2))
                loss_x = logloss_x + opt.gamma * quantization_x + opt.eta * balance_x
                loss_x = loss_x / num_train

            # 更新 F_buffer（在 autocast 外，确保 float32）
            F_buffer[indices, :] = cur_f.float().data

            optimizer_img.zero_grad()
            if use_amp:
                scaler.scale(loss_x).backward()
                scaler.unscale_(optimizer_img)
                torch.nn.utils.clip_grad_norm_(img_model.parameters(), max_norm=5.0)
                scaler.step(optimizer_img)
                scaler.update()
            else:
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

            # 混合精度训练
            with autocast('cuda', enabled=use_amp):
                cur_g = txt_model(text)

                theta_y = 1.0 / 2 * torch.matmul(cur_g, F_buffer.t())
                logloss_y = -torch.sum(S * theta_y - torch_func.softplus(theta_y))
                quantization_y = torch.sum(torch.pow(B[ind, :] - cur_g, 2))
                balance_y = torch.sum(torch.pow(cur_g.t().mm(ones) + G_buffer[unupdated_ind].t().mm(ones_), 2))
                loss_y = logloss_y + opt.gamma * quantization_y + opt.eta * balance_y
                loss_y = loss_y / num_train

            # 更新 G_buffer（在 autocast 外，确保 float32）
            G_buffer[ind, :] = cur_g.float().data

            optimizer_txt.zero_grad()
            if use_amp:
                scaler.scale(loss_y).backward()
                scaler.unscale_(optimizer_txt)
                torch.nn.utils.clip_grad_norm_(txt_model.parameters(), max_norm=5.0)
                scaler.step(optimizer_txt)
                scaler.update()
            else:
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
        if opt.valid_interval > 0 and (epoch + 1) % opt.valid_interval == 0:
            # 生成查询和检索集的哈希码
            qBX = generate_image_code_from_loader(img_model, query_img_loader, opt.bit, opt.use_gpu)
            qBY = generate_text_code_from_dataset(txt_model, query_txt_dataset, opt.bit, opt.use_gpu)
            rBX = generate_image_code_from_loader(img_model, retrieval_img_loader, opt.bit, opt.use_gpu)
            rBY = generate_text_code_from_dataset(txt_model, retrieval_txt_dataset, opt.bit, opt.use_gpu)

            mapi2t = calc_map_k(qBX, rBY, query_L, retrieval_L)
            mapt2i = calc_map_k(qBY, rBX, query_L, retrieval_L)

            print('...epoch: %3d, valid MAP: MAP(i->t): %3.4f, MAP(t->i): %3.4f' %
                  (epoch + 1, mapi2t, mapt2i))

            # 记录 mAP 历史
            result['mAP_history'].append({
                'epoch': epoch + 1,
                'mapi2t': float(mapi2t),
                'mapt2i': float(mapt2i)
            })

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
    if opt.valid_interval == 0:
        print("执行最终验证...")
        qBX = generate_image_code_from_loader(img_model, query_img_loader, opt.bit, opt.use_gpu)
        qBY = generate_text_code_from_dataset(txt_model, query_txt_dataset, opt.bit, opt.use_gpu)
        rBX = generate_image_code_from_loader(img_model, retrieval_img_loader, opt.bit, opt.use_gpu)
        rBY = generate_text_code_from_dataset(txt_model, retrieval_txt_dataset, opt.bit, opt.use_gpu)

        mapi2t = calc_map_k(qBX, rBY, query_L, retrieval_L)
        mapt2i = calc_map_k(qBY, rBX, query_L, retrieval_L)

        print('   Final MAP: MAP(i->t): %3.4f, MAP(t->i): %3.4f' % (mapi2t, mapt2i))
        max_mapi2t = mapi2t
        max_mapt2i = mapt2i

        # 记录最终验证结果
        result['mAP_history'].append({
            'epoch': opt.max_epoch,
            'mapi2t': float(mapi2t),
            'mapt2i': float(mapt2i)
        })

        save_model(img_model, 'img_model.pth', result_dir=result_dir, use_gpu=opt.use_gpu)
        save_model(txt_model, 'txt_model.pth', result_dir=result_dir, use_gpu=opt.use_gpu)

    print('   max MAP: MAP(i->t): %3.4f, MAP(t->i): %3.4f' % (max_mapi2t, max_mapt2i))
    result['mapi2t'] = max_mapi2t
    result['mapt2i'] = max_mapt2i

    write_result(result, result_dir)


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
    计算 DCMH 总损失。

    注意：不在此处归一化，梯度归一化在 backward() 调用时进行，
    以匹配 Matlab 原始实现的梯度量级。

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

    # 不归一化，配合 backward 时的 (loss / num_train)
    loss = term1 + gamma * term2 + eta * term3
    return loss


def generate_image_code_from_loader(img_model, loader, bit, use_gpu):
    """从 DataLoader 生成图像哈希码"""
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


def generate_text_code_from_dataset(txt_model, dataset, bit, use_gpu, batch_size=128):
    """
    从 Dataset 批量生成文本哈希码（GPU 加速版本）。

    使用 DataLoader 批处理代替逐样本处理，
    减少 GPU 调用次数，大幅提升推理速度。

    参数：
        txt_model: 文本模型
        dataset: DCMHTextDataset 实例
        bit: 哈希码位数
        use_gpu: 是否使用 GPU
        batch_size: 批次大小（默认 128）

    返回：
        B: 哈希码矩阵 [num_data, bit]
    """
    num_data = len(dataset)
    B = torch.zeros(num_data, bit, dtype=torch.float)
    if use_gpu and torch.cuda.is_available():
        B = B.cuda()

    txt_model.eval()
    # 创建 DataLoader 进行批处理
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                       num_workers=0, pin_memory=True)

    with torch.no_grad():
        for tags, labels, indices in tqdm(loader, desc='Generating text codes'):
            # tags: [batch, 1, y_dim, 1]
            text = tags.type(torch.float)
            if use_gpu and torch.cuda.is_available():
                text = text.cuda()
            cur_g = txt_model(text)  # 批量推理
            B[indices, :] = cur_g.data

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
            python train_dcmh.py train --valid_interval=5
            python train_dcmh.py train --valid_interval=0  # 仅最终验证
            python train_dcmh.py train --resume_from=results/flickr-25k/20260323_120612
            python train_dcmh.py help
    available args:''')
    for k, v in TrainConfig.__dict__.items():
        if not k.startswith('__') and not callable(v):
            print('\t\t{0}: {1}'.format(k, v))


if __name__ == '__main__':
    import fire
    fire.Fire()
