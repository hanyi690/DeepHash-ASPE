"""
DCMH 训练后处理脚本

功能：
1. 读取训练生成的 result.txt 文件
2. 绘制训练曲线图
3. 执行 ASPE 评估
4. 生成综合报告

使用方式：
    # 仅绘制训练曲线
    python training/postprocess.py plot --result-dir results/flickr-25k

    # 仅 ASPE 评估
    python training/postprocess.py aspe --result-dir results/flickr-25k --bit 64

    # 完整后处理（曲线 + ASPE + 报告）
    python training/postprocess.py all --result-dir results/flickr-25k --bit 64

    # 从模型重新生成哈希码并评估
    python training/postprocess.py eval --model-dir results/flickr-25k --data data/FLICKR-25K.mat --bit 64
"""

import os
import sys
import json
import argparse
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

# 添加父目录到路径以便导入
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
from tqdm import tqdm

# 导入本地模块
from config.dcmh_config import DCMHConfig
from reference.DCMH.models.img_module import ImgModule
from reference.DCMH.models.txt_module import TxtModule
from core.hashing.dcmh_data_loader import load_data, split_data, load_pretrain_model
from core.retrieval.dcmh_metrics import calc_map_k, calc_hammingDist
from core.aspe.dcmh_wrapper import ASPEForDCMH

# 默认配置
training_opt = DCMHConfig()


# ============================================================================
# 辅助函数
# ============================================================================

def get_y_dim(data_path: str) -> int:
    """
    从数据集获取标签维度

    参数：
        data_path: 数据集路径

    返回：
        标签维度 (y_dim)
    """
    try:
        _, tags, _ = load_data(data_path)
        return tags.shape[1]
    except Exception as e:
        print(f"警告：无法从数据集获取 y_dim: {e}")
        return 1386  # FLICKR-25K 默认值


# ============================================================================
# 功能模块 1: 读取训练历史
# ============================================================================

def load_training_history(result_dir: str) -> Dict[str, Any]:
    """
    读取 result.txt 文件

    result.txt 格式:
    loss [val1, val2, ...]
    mapi2t 0.xxxx 或 tensor(0.xxxx, device='cuda:0')
    mapt2i 0.xxxx 或 tensor(0.xxxx, device='cuda:0')

    参数：
        result_dir: 结果目录路径

    返回：
        包含训练历史的字典
    """
    import re

    result_path = os.path.join(result_dir, 'result.txt')

    if not os.path.exists(result_path):
        raise FileNotFoundError(f"找不到结果文件：{result_path}")

    history = {
        'loss': [],
        'mapi2t': None,
        'mapt2i': None
    }

    def extract_float(value_str: str) -> float:
        """从字符串中提取浮点数，支持 tensor 格式"""
        # 尝试匹配 tensor(x.xxx) 格式
        match = re.search(r'tensor\(([0-9.]+)', value_str)
        if match:
            return float(match.group(1))
        # 尝试直接转换为浮点数
        try:
            return float(value_str)
        except ValueError:
            raise ValueError(f"无法解析数值: {value_str}")

    with open(result_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split(None, 1)  # 分割成两部分：键和值
            if len(parts) < 2:
                continue

            key = parts[0]
            value = parts[1]

            if key == 'loss':
                # 解析列表格式：[val1, val2, ...]
                value = value.strip('[]')
                if value:
                    history['loss'] = [float(v.strip()) for v in value.split(',')]
            elif key == 'mapi2t':
                history['mapi2t'] = extract_float(value)
            elif key == 'mapt2i':
                history['mapt2i'] = extract_float(value)

    return history


# ============================================================================
# 功能模块 2: 绘制训练曲线
# ============================================================================

def plot_training_curves(history: Dict[str, Any], output_dir: str) -> Dict[str, str]:
    """
    生成训练曲线可视化图

    参数：
        history: 训练历史数据
        output_dir: 输出目录

    返回：
        生成的文件路径字典
    """
    os.makedirs(output_dir, exist_ok=True)

    output_files = {}

    # 设置中文字体（如果可用）
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    # 1. 绘制 Loss 曲线
    if history.get('loss') and len(history['loss']) > 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        epochs = range(1, len(history['loss']) + 1)
        ax.plot(epochs, history['loss'], 'b-', linewidth=2, marker='o', markersize=3)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Training Loss Curve', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(left=1)

        plt.tight_layout()
        loss_path = os.path.join(output_dir, 'loss_curve.png')
        plt.savefig(loss_path, dpi=150, bbox_inches='tight')
        plt.close()
        output_files['loss_curve'] = loss_path

    # 2. 绘制 mAP 曲线（如果有多个 epoch 的数据）
    if history.get('mapi2t') is not None and history.get('mapt2i') is not None:
        fig, ax = plt.subplots(figsize=(10, 6))

        # 如果只有一个最终值，绘制柱状图
        if len(history.get('loss', [])) <= 1:
            categories = ['mAP(i→t)', 'mAP(t→i)']
            values = [history['mapi2t'], history['mapt2i']]
            bars = ax.bar(categories, values, color=['steelblue', 'coral'], alpha=0.7)
            ax.set_ylabel('mAP', fontsize=12)
            ax.set_title('Final mAP Results', fontsize=14)
            ax.set_ylim(0, 1)

            # 在柱子上方添加数值标签
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                       f'{val:.4f}', ha='center', va='bottom', fontsize=11)
        else:
            # 如果有多个 epoch，绘制曲线（假设每个 epoch 都有 mAP 记录）
            epochs = range(1, len(history['loss']) + 1)
            ax.plot(epochs, [history['mapi2t']] * len(epochs), 'b-',
                   linewidth=2, marker='o', markersize=3, label='mAP(i→t)')
            ax.plot(epochs, [history['mapt2i']] * len(epochs), 'r-',
                   linewidth=2, marker='s', markersize=3, label='mAP(t→i)')
            ax.set_xlabel('Epoch', fontsize=12)
            ax.set_ylabel('mAP', fontsize=12)
            ax.set_title('Training mAP Curves', fontsize=14)
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        map_path = os.path.join(output_dir, 'map_curve.png')
        plt.savefig(map_path, dpi=150, bbox_inches='tight')
        plt.close()
        output_files['map_curve'] = map_path

    # 3. 绘制综合图
    if history.get('loss') and len(history['loss']) > 0:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Loss 曲线
        epochs = range(1, len(history['loss']) + 1)
        ax1.plot(epochs, history['loss'], 'b-', linewidth=2, marker='o', markersize=3)
        ax1.set_xlabel('Epoch', fontsize=11)
        ax1.set_ylabel('Loss', fontsize=11)
        ax1.set_title('Training Loss', fontsize=13)
        ax1.grid(True, alpha=0.3)

        # mAP 柱状图
        categories = ['mAP(i→t)', 'mAP(t→i)']
        values = [history.get('mapi2t', 0), history.get('mapt2i', 0)]
        colors = ['steelblue', 'coral']
        bars = ax2.bar(categories, values, color=colors, alpha=0.7)
        ax2.set_ylabel('mAP', fontsize=11)
        ax2.set_title('Final mAP Results', fontsize=13)
        ax2.set_ylim(0, 1)

        for bar, val in zip(bars, values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=10)

        plt.tight_layout()
        combined_path = os.path.join(output_dir, 'training_curves.png')
        plt.savefig(combined_path, dpi=150, bbox_inches='tight')
        plt.close()
        output_files['combined'] = combined_path

    return output_files


# ============================================================================
# 功能模块 3: 加载模型
# ============================================================================

def load_models(result_dir: str, bit: int = 64, y_dim: int = 1386,
                use_gpu: bool = True) -> Tuple[Optional[ImgModule], Optional[TxtModule]]:
    """
    从.pth 文件加载模型权重

    参数：
        result_dir: 结果目录
        bit: 哈希码位数
        y_dim: 标签维度
        use_gpu: 是否使用 GPU

    返回：
        (img_model, txt_model) 元组
    """
    img_model_path = os.path.join(result_dir, 'img_model.pth')
    txt_model_path = os.path.join(result_dir, 'txt_model.pth')

    img_model = None
    txt_model = None

    # 加载预训练模型（用于初始化）
    pretrain_model = None
    try:
        pretrain_path = training_opt.pretrain_model_path
        if os.path.exists(pretrain_path):
            pretrain_model = load_pretrain_model(pretrain_path)
    except Exception as e:
        print(f"警告：无法加载预训练模型：{e}")

    # 加载图像模型
    if os.path.exists(img_model_path):
        img_model = ImgModule(bit, pretrain_model)
        try:
            img_model.load(img_model_path, use_gpu=use_gpu)
            print(f"成功加载图像模型：{img_model_path}")
        except Exception as e:
            print(f"加载图像模型失败：{e}")
            img_model = None
    else:
        print(f"警告：未找到图像模型文件：{img_model_path}")

    # 加载文本模型
    if os.path.exists(txt_model_path):
        txt_model = TxtModule(y_dim, bit)
        try:
            txt_model.load(txt_model_path, use_gpu=use_gpu)
            print(f"成功加载文本模型：{txt_model_path}")
        except Exception as e:
            print(f"加载文本模型失败：{e}")
            txt_model = None
    else:
        print(f"警告：未找到文本模型文件：{txt_model_path}")

    if use_gpu and torch.cuda.is_available():
        if img_model:
            img_model = img_model.cuda()
        if txt_model:
            txt_model = txt_model.cuda()
    elif use_gpu:
        print("警告：请求使用 GPU，但 CUDA 不可用，将使用 CPU")

    return img_model, txt_model


# ============================================================================
# 功能模块 4: 生成哈希码
# ============================================================================

def generate_image_code(model: ImgModule, X: torch.Tensor, bit: int,
                        batch_size: int = 64, use_gpu: bool = True) -> torch.Tensor:
    """
    生成图像哈希码（复制自 train_flickr25k.py）
    """
    # 确定是否真正使用 GPU
    actual_use_gpu = use_gpu and torch.cuda.is_available()
    if use_gpu and not actual_use_gpu:
        print("警告：请求使用 GPU，但 CUDA 不可用，将使用 CPU")

    num_data = X.shape[0]
    index = np.linspace(0, num_data - 1, num_data).astype(int)
    B = torch.zeros(num_data, bit, dtype=torch.float)

    if actual_use_gpu:
        B = B.cuda()
        model = model.cuda()

    model.eval()
    with torch.no_grad():
        for i in tqdm(range(num_data // batch_size + 1), desc="生成图像哈希码"):
            ind = index[i * batch_size: min((i + 1) * batch_size, num_data)]
            image = X[ind].type(torch.float)
            if actual_use_gpu:
                image = image.cuda()
            cur_f = model(image)
            B[ind, :] = cur_f.data

    B = torch.sign(B)
    return B


def generate_text_code(model: TxtModule, Y: torch.Tensor, bit: int,
                       batch_size: int = 64, use_gpu: bool = True) -> torch.Tensor:
    """
    生成文本哈希码（复制自 train_flickr25k.py）
    """
    # 确定是否真正使用 GPU
    actual_use_gpu = use_gpu and torch.cuda.is_available()
    if use_gpu and not actual_use_gpu:
        print("警告：请求使用 GPU，但 CUDA 不可用，将使用 CPU")

    num_data = Y.shape[0]
    index = np.linspace(0, num_data - 1, num_data).astype(int)
    B = torch.zeros(num_data, bit, dtype=torch.float)

    if actual_use_gpu:
        B = B.cuda()
        model = model.cuda()

    model.eval()
    with torch.no_grad():
        for i in tqdm(range(num_data // batch_size + 1), desc="生成文本哈希码"):
            ind = index[i * batch_size: min((i + 1) * batch_size, num_data)]
            text = Y[ind].unsqueeze(1).unsqueeze(-1).type(torch.float)
            if actual_use_gpu:
                text = text.cuda()
            cur_g = model(text)
            B[ind, :] = cur_g.data

    B = torch.sign(B)
    return B


# ============================================================================
# 功能模块 5: ASPE 评估
# ============================================================================

def run_aspe_evaluation(
    img_model: Optional[ImgModule],
    txt_model: Optional[TxtModule],
    data_path: str,
    bit: int,
    output_dir: str,
    batch_size: int = 64,
    use_gpu: bool = True,
    aspe_seed: int = 42
) -> Dict[str, Any]:
    """
    执行 ASPE 评估

    参数：
        img_model: 图像模型
        txt_model: 文本模型
        data_path: 数据集路径
        bit: 哈希码位数
        output_dir: 输出目录
        batch_size: batch 大小
        use_gpu: 是否使用 GPU
        aspe_seed: ASPE 随机种子

    返回：
        评估结果字典
    """
    # 确定是否真正使用 GPU
    actual_use_gpu = use_gpu and torch.cuda.is_available()
    if use_gpu and not actual_use_gpu:
        print("警告：请求使用 GPU，但 CUDA 不可用，将使用 CPU")

    os.makedirs(output_dir, exist_ok=True)

    results = {
        'bit': bit,
        'plaintext': {},
        'ciphertext': {},
        'consistency_verified': False
    }

    # 加载数据
    print("加载数据集...")
    try:
        images, tags, labels = load_data(data_path)
    except Exception as e:
        print(f"加载数据失败：{e}")
        return results

    y_dim = tags.shape[1]

    # 使用标准划分
    X, Y, L = split_data(images, tags, labels)

    query_L = torch.from_numpy(L['query'])
    query_x = torch.from_numpy(X['query'])
    query_y = torch.from_numpy(Y['query'])

    retrieval_L = torch.from_numpy(L['retrieval'])
    retrieval_x = torch.from_numpy(X['retrieval'])
    retrieval_y = torch.from_numpy(Y['retrieval'])

    if actual_use_gpu:
        query_L = query_L.cuda()
        retrieval_L = retrieval_L.cuda()

    # 生成哈希码
    print("生成哈希码...")

    if img_model is not None:
        qBX = generate_image_code(img_model, query_x, bit, batch_size, use_gpu)
        rBX = generate_image_code(img_model, retrieval_x, bit, batch_size, use_gpu)
        print(f"图像哈希码：query={qBX.shape}, retrieval={rBX.shape}")
    else:
        qBX = None
        rBX = None

    if txt_model is not None:
        qBY = generate_text_code(txt_model, query_y, bit, batch_size, use_gpu)
        rBY = generate_text_code(txt_model, retrieval_y, bit, batch_size, use_gpu)
        print(f"文本哈希码：query={qBY.shape}, retrieval={rBY.shape}")
    else:
        qBY = None
        rBY = None

    # 明文 mAP 评估
    print("\n明文 mAP 评估...")

    if qBX is not None and rBY is not None:
        map_i2t = calc_map_k(qBX, rBY, query_L, retrieval_L)
        results['plaintext']['map_i2t'] = float(map_i2t.cpu().item() if isinstance(map_i2t, torch.Tensor) else map_i2t)
        print(f"  mAP(i→t): {results['plaintext']['map_i2t']:.4f}")

    if qBY is not None and rBX is not None:
        map_t2i = calc_map_k(qBY, rBX, query_L, retrieval_L)
        results['plaintext']['map_t2i'] = float(map_t2i.cpu().item() if isinstance(map_t2i, torch.Tensor) else map_t2i)
        print(f"  mAP(t→i): {results['plaintext']['map_t2i']:.4f}")

    # ASPE 密文评估
    print("\nASPE 密文评估...")

    aspe = ASPEForDCMH(bit_dim=bit, seed=aspe_seed)

    # 加密检索库
    if rBX is not None:
        print("  加密检索库图像哈希码...")
        encrypted_rBX = aspe.GenEnc(rBX)
    else:
        encrypted_rBX = None

    if rBY is not None:
        print("  加密检索库文本哈希码...")
        encrypted_rBY = aspe.GenEnc(rBY)
    else:
        encrypted_rBY = None

    # 生成查询陷阱门并评估
    if qBX is not None and encrypted_rBY is not None:
        print("  计算密文 mAP(i→t)...")
        encrypted_qBX = aspe.GenTrap(qBX)
        map_enc_i2t = aspe.calc_ciphertext_map(
            encrypted_qBX, encrypted_rBY,
            query_L.cpu().numpy(), retrieval_L.cpu().numpy()
        )
        results['ciphertext']['map_i2t'] = float(map_enc_i2t)
        print(f"  mAP(i→t): {results['ciphertext']['map_i2t']:.4f}")

    if qBY is not None and encrypted_rBX is not None:
        print("  计算密文 mAP(t→i)...")
        encrypted_qBY = aspe.GenTrap(qBY)
        map_enc_t2i = aspe.calc_ciphertext_map(
            encrypted_qBY, encrypted_rBX,
            query_L.cpu().numpy(), retrieval_L.cpu().numpy()
        )
        results['ciphertext']['map_t2i'] = float(map_enc_t2i)
        print(f"  mAP(t→i): {results['ciphertext']['map_t2i']:.4f}")

    # 验证排序一致性
    print("\n验证 ASPE 排序一致性...")
    if rBX is not None and qBX is not None:
        consistent = aspe.verify_sorting_consistency(
            qBX.cpu().numpy()[:10],
            rBX.cpu().numpy()[:100]
        )
        results['consistency_verified'] = consistent
        print(f"  排序一致性：{'通过' if consistent else '失败'}")

    # 保存结果
    output_path = os.path.join(output_dir, 'aspe_evaluation.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nASPE 评估结果已保存：{output_path}")

    return results


# ============================================================================
# 功能模块 6: 生成报告
# ============================================================================

def plot_aspe_comparison(aspe_results: Dict[str, Any], output_dir: str) -> str:
    """
    生成明文 vs 密文 mAP 对比图

    参数：
        aspe_results: ASPE 评估结果
        output_dir: 输出目录

    返回：
        图表文件路径
    """
    os.makedirs(output_dir, exist_ok=True)

    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    labels = ['mAP(i→t)', 'mAP(t→i)']
    plaintext = [
        aspe_results.get('plaintext', {}).get('map_i2t', 0),
        aspe_results.get('plaintext', {}).get('map_t2i', 0)
    ]
    ciphertext = [
        aspe_results.get('ciphertext', {}).get('map_i2t', 0),
        aspe_results.get('ciphertext', {}).get('map_t2i', 0)
    ]

    x = range(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar([i - width/2 for i in x], plaintext, width, label='明文', color='#1f77b4')
    bars2 = ax.bar([i + width/2 for i in x], ciphertext, width, label='密文(ASPE)', color='#ff7f0e')

    ax.set_ylabel('mAP', fontsize=12)
    ax.set_title(f'ASPE 加密前后 mAP 对比 ({aspe_results.get("bit", 64)} bits)', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1)

    # 添加数值标签
    for bar, val in zip(bars1, plaintext):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
               f'{val:.4f}', ha='center', va='bottom', fontsize=10)
    for bar, val in zip(bars2, ciphertext):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
               f'{val:.4f}', ha='center', va='bottom', fontsize=10)

    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'aspe_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"ASPE 对比图已生成：{output_path}")
    return output_path


def generate_report(
    history: Dict[str, Any],
    aspe_results: Optional[Dict[str, Any]],
    plot_files: Dict[str, str],
    output_dir: str,
    config: Optional[Dict[str, Any]] = None
) -> str:
    """
    生成 Markdown 格式的综合报告

    参数：
        history: 训练历史
        aspe_results: ASPE 评估结果
        plot_files: 图表文件路径
        output_dir: 输出目录
        config: 配置信息

    返回：
        报告文件路径
    """
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    report_lines = [
        "# DCMH 训练后处理报告",
        "",
        f"生成时间：{timestamp}",
        "",
        "## 1. 训练配置",
        "",
    ]

    if config:
        for key, value in config.items():
            report_lines.append(f"- **{key}**: {value}")
    else:
        report_lines.extend([
            f"- 哈希码位数 (bit): {history.get('bit', 'N/A')}",
            f"- 训练轮数 (epochs): {len(history.get('loss', []))}",
        ])

    report_lines.extend([
        "",
        "## 2. 训练历史",
        "",
        "### 2.1 Loss 曲线",
        "",
    ])

    if 'loss_curve' in plot_files:
        report_lines.append(f"![Loss Curve]({os.path.basename(plot_files['loss_curve'])})")
        report_lines.append("")
        report_lines.append(f"- 初始 Loss: {history['loss'][0]:.4f}" if history.get('loss') else "- 初始 Loss: N/A")
        report_lines.append(f"- 最终 Loss: {history['loss'][-1]:.4f}" if history.get('loss') else "- 最终 Loss: N/A")

    report_lines.extend([
        "",
        "### 2.2 mAP 结果",
        "",
    ])

    if 'map_curve' in plot_files:
        report_lines.append(f"![mAP Curve]({os.path.basename(plot_files['map_curve'])})")
        report_lines.append("")

    if history.get('mapi2t') is not None:
        report_lines.append(f"- **mAP(i→t)**: {history['mapi2t']:.4f}")
    if history.get('mapt2i') is not None:
        report_lines.append(f"- **mAP(t→i)**: {history['mapt2i']:.4f}")

    report_lines.extend([
        "",
        "## 3. ASPE 评估",
        "",
    ])

    if aspe_results:
        report_lines.extend([
            "",
            "### 3.1 明文检索性能",
            "",
        ])
        if 'map_i2t' in aspe_results.get('plaintext', {}):
            report_lines.append(f"- mAP(i→t): {aspe_results['plaintext']['map_i2t']:.4f}")
        if 'map_t2i' in aspe_results.get('plaintext', {}):
            report_lines.append(f"- mAP(t→i): {aspe_results['plaintext']['map_t2i']:.4f}")

        report_lines.extend([
            "",
            "### 3.2 密文检索性能",
            "",
        ])
        if 'map_i2t' in aspe_results.get('ciphertext', {}):
            report_lines.append(f"- mAP(i→t): {aspe_results['ciphertext']['map_i2t']:.4f}")
        if 'map_t2i' in aspe_results.get('ciphertext', {}):
            report_lines.append(f"- mAP(t→i): {aspe_results['ciphertext']['map_t2i']:.4f}")

        report_lines.extend([
            "",
            "### 3.3 排序一致性验证",
            "",
            f"- 验证结果：{'通过' if aspe_results.get('consistency_verified') else '失败'}",
        ])

        # 添加 ASPE 对比图
        if 'aspe_comparison' in plot_files:
            report_lines.extend([
                "",
                "### 3.4 明文 vs 密文 mAP 对比图",
                "",
                f"![ASPE Comparison]({os.path.basename(plot_files['aspe_comparison'])})",
            ])
    else:
        report_lines.append("*未执行 ASPE 评估*")

    report_lines.extend([
        "",
        "## 4. 结论",
        "",
    ])

    if aspe_results and aspe_results.get('consistency_verified'):
        report_lines.append(
            "ASPE 加密方案成功保持了哈希码的排序关系，可以在密文状态下实现与明文相同的检索精度。"
        )
    elif aspe_results:
        report_lines.append(
            "ASPE 评估已完成，请检查上述结果以验证加密方案的有效性。"
        )
    else:
        report_lines.append(
            "请运行 ASPE 评估以获取完整的分析报告。"
        )

    report_lines.extend([
        "",
        "---",
        "",
        "*本报告由 DCMH 后处理脚本自动生成*",
    ])

    # 写入报告文件
    report_path = os.path.join(output_dir, 'report.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))

    print(f"综合报告已生成：{report_path}")
    return report_path


# ============================================================================
# 命令行接口
# ============================================================================

def cmd_plot(args):
    """仅绘制训练曲线"""
    print("=" * 50)
    print("DCMH 训练曲线绘制")
    print("=" * 50)

    # 加载训练历史
    print(f"\n加载训练历史：{args.result_dir}")
    history = load_training_history(args.result_dir)

    print(f"  - Loss 记录数：{len(history.get('loss', []))}")
    print(f"  - mAP(i→t): {history.get('mapi2t', 'N/A')}")
    print(f"  - mAP(t→i): {history.get('mapt2i', 'N/A')}")

    # 绘制曲线
    print(f"\n绘制训练曲线，输出到：{args.output_dir}")
    plot_files = plot_training_curves(history, args.output_dir)

    print("\n生成的文件:")
    for name, path in plot_files.items():
        print(f"  - {path}")

    print("\n完成!")


def cmd_aspe(args):
    """仅 ASPE 评估"""
    print("=" * 50)
    print("DCMH ASPE 评估")
    print("=" * 50)

    # 从数据集获取 y_dim
    y_dim = get_y_dim(args.data_path)
    print(f"\n标签维度 (y_dim): {y_dim}")

    # 加载模型
    print(f"\n加载模型：{args.result_dir}")
    img_model, txt_model = load_models(
        args.result_dir, bit=args.bit, y_dim=y_dim, use_gpu=args.gpu
    )

    # 运行 ASPE 评估
    print(f"\n执行 ASPE 评估，数据集：{args.data_path}")
    aspe_results = run_aspe_evaluation(
        img_model=img_model,
        txt_model=txt_model,
        data_path=args.data_path,
        bit=args.bit,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        use_gpu=args.gpu,
        aspe_seed=args.aspe_seed
    )

    print("\n完成!")


def cmd_all(args):
    """完整后处理（曲线 + ASPE + 报告）"""
    print("=" * 50)
    print("DCMH 完整后处理")
    print("=" * 50)

    # 确定 y_dim（从数据集获取）
    if os.path.exists(args.data_path):
        y_dim = get_y_dim(args.data_path)
        print(f"标签维度 (y_dim): {y_dim}")
    else:
        print(f"警告：数据集不存在：{args.data_path}，使用默认值 y_dim=1386")
        y_dim = 1386

    # 1. 加载训练历史
    print(f"\n[1/4] 加载训练历史：{args.result_dir}")
    history = load_training_history(args.result_dir)
    history['bit'] = args.bit

    print(f"  - Loss 记录数：{len(history.get('loss', []))}")
    print(f"  - mAP(i→t): {history.get('mapi2t', 'N/A')}")
    print(f"  - mAP(t→i): {history.get('mapt2i', 'N/A')}")

    # 2. 绘制曲线
    print(f"\n[2/4] 绘制训练曲线")
    plot_files = plot_training_curves(history, args.output_dir)
    print(f"  生成 {len(plot_files)} 个图表文件")

    # 3. 加载模型并运行 ASPE 评估
    print(f"\n[3/4] 执行 ASPE 评估")

    # 检查数据路径
    if not os.path.exists(args.data_path):
        print(f"  警告：数据集不存在：{args.data_path}")
        print("  跳过 ASPE 评估，仅生成训练曲线报告")
        aspe_results = None
    else:
        img_model, txt_model = load_models(
            args.result_dir, bit=args.bit, y_dim=y_dim, use_gpu=args.gpu
        )

        if img_model is not None or txt_model is not None:
            aspe_results = run_aspe_evaluation(
                img_model=img_model,
                txt_model=txt_model,
                data_path=args.data_path,
                bit=args.bit,
                output_dir=args.output_dir,
                batch_size=args.batch_size,
                use_gpu=args.gpu,
                aspe_seed=args.aspe_seed
            )
            # 生成 ASPE 对比图
            if aspe_results and aspe_results.get('plaintext') and aspe_results.get('ciphertext'):
                aspe_comparison_path = plot_aspe_comparison(aspe_results, args.output_dir)
                plot_files['aspe_comparison'] = aspe_comparison_path
        else:
            print("  警告：未加载到任何模型，跳过 ASPE 评估")
            aspe_results = None

    # 4. 生成报告
    print(f"\n[4/4] 生成综合报告")

    # 构建配置信息
    config = {
        'bit': args.bit,
        'epochs': len(history.get('loss', [])),
        'batch_size': args.batch_size,
        'result_dir': args.result_dir,
        'data_path': args.data_path,
    }

    report_path = generate_report(
        history=history,
        aspe_results=aspe_results,
        plot_files=plot_files,
        output_dir=args.output_dir,
        config=config
    )

    print("\n" + "=" * 50)
    print("后处理完成!")
    print("=" * 50)
    print(f"\n输出目录：{args.output_dir}")
    print("\n生成的文件:")
    for name, path in plot_files.items():
        print(f"  - {path}")
    if aspe_results:
        print(f"  - {os.path.join(args.output_dir, 'aspe_evaluation.json')}")
    print(f"  - {report_path}")


def cmd_eval(args):
    """从模型重新评估"""
    print("=" * 50)
    print("DCMH 模型评估")
    print("=" * 50)

    # 从数据集获取 y_dim
    y_dim = get_y_dim(args.data_path)
    print(f"\n标签维度 (y_dim): {y_dim}")

    # 加载模型
    print(f"\n加载模型：{args.model_dir}")
    img_model, txt_model = load_models(
        args.model_dir, bit=args.bit, y_dim=y_dim, use_gpu=args.gpu
    )

    if img_model is None and txt_model is None:
        print("错误：未能加载任何模型")
        return

    # 运行 ASPE 评估
    print(f"\n执行评估，数据集：{args.data_path}")
    aspe_results = run_aspe_evaluation(
        img_model=img_model,
        txt_model=txt_model,
        data_path=args.data_path,
        bit=args.bit,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        use_gpu=args.gpu,
        aspe_seed=args.aspe_seed
    )

    # 生成 ASPE 对比图
    if aspe_results and aspe_results.get('plaintext') and aspe_results.get('ciphertext'):
        plot_aspe_comparison(aspe_results, args.output_dir)

    print("\n完成!")


def main():
    parser = argparse.ArgumentParser(
        description='DCMH 训练后处理工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s plot --result-dir results/flickr-25k
  %(prog)s aspe --result-dir results/flickr-25k --bit 64 --data data/FLICKR-25K.mat
  %(prog)s all --result-dir results/flickr-25k --bit 64 --data data/FLICKR-25K.mat
  %(prog)s eval --model-dir results/flickr-25k --bit 64 --data data/FLICKR-25K.mat
        """
    )

    parser.add_argument('--gpu', type=bool, default=True,
                       help='是否使用 GPU (默认：True)')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='batch size (默认：64)')
    parser.add_argument('--aspe-seed', type=int, default=42,
                       help='ASPE 随机种子 (默认：42)')

    subparsers = parser.add_subparsers(dest='command', help='命令')

    # plot 命令
    plot_parser = subparsers.add_parser('plot', help='仅绘制训练曲线')
    plot_parser.add_argument('--result-dir', type=str, required=True,
                            help='训练结果目录')
    plot_parser.add_argument('--output-dir', type=str, default='results/postprocess/',
                            help='输出目录')
    plot_parser.set_defaults(func=cmd_plot)

    # aspe 命令
    aspe_parser = subparsers.add_parser('aspe', help='仅 ASPE 评估')
    aspe_parser.add_argument('--result-dir', type=str, required=True,
                            help='训练结果目录')
    aspe_parser.add_argument('--bit', type=int, default=64,
                            help='哈希码位数')
    aspe_parser.add_argument('--data-path', type=str, default='data/FLICKR-25K.mat',
                            help='数据集路径')
    aspe_parser.add_argument('--output-dir', type=str, default='results/postprocess/',
                            help='输出目录')
    aspe_parser.set_defaults(func=cmd_aspe)

    # all 命令
    all_parser = subparsers.add_parser('all', help='完整后处理（曲线 + ASPE + 报告）')
    all_parser.add_argument('--result-dir', type=str, required=True,
                           help='训练结果目录')
    all_parser.add_argument('--bit', type=int, default=64,
                           help='哈希码位数')
    all_parser.add_argument('--data-path', type=str, default='data/FLICKR-25K.mat',
                           help='数据集路径')
    all_parser.add_argument('--output-dir', type=str, default='results/postprocess/',
                           help='输出目录')
    all_parser.set_defaults(func=cmd_all)

    # eval 命令
    eval_parser = subparsers.add_parser('eval', help='从模型重新评估')
    eval_parser.add_argument('--model-dir', type=str, required=True,
                            help='模型目录')
    eval_parser.add_argument('--bit', type=int, default=64,
                            help='哈希码位数')
    eval_parser.add_argument('--data-path', type=str, default='data/FLICKR-25K.mat',
                            help='数据集路径')
    eval_parser.add_argument('--output-dir', type=str, default='results/postprocess/',
                            help='输出目录')
    eval_parser.set_defaults(func=cmd_eval)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    args.func(args)


if __name__ == '__main__':
    main()
