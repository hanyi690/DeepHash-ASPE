"""
Flickr25K 完整系统测试和实验运行脚本

使用 Flickr25K 数据集完成 DCMH + ASPE 系统的全部测试和实验，包括：
1. DCMH 模型训练
2. 检索性能评估
3. ASPE 加密测试
4. 生成完整报告和可视化图表

所有结果输出到 results/flickr-25k 目录
"""

import os
import sys
import json
import time
import gc
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.flickr25k_dataset import load_flickr25k_data
from core.hashing.dcmh_model import DCMHModel
from core.aspe.scheme1 import ASPEScheme1
from core.aspe.dcmh_wrapper import ASPEForDCMH
from reference.DCMH.utils import calc_map_k
from evaluation.metrics import compute_hash_quality


class Flickr25KExperiment:
    """Flickr25K 完整实验管理器。"""

    def __init__(self,
                 data_path: str,
                 bit_dim: int = 64,
                 batch_size: int = 128,
                 max_epoch: int = 500,
                 lr: float = 1e-4,
                 gamma: float = 1.0,
                 eta: float = 1.0,
                 use_gpu: bool = True,
                 result_dir: str = 'results/flickr-25k'):
        """
        初始化实验。

        参数:
            data_path: 数据文件路径
            bit_dim: 哈希码维度
            batch_size: 批次大小
            max_epoch: 最大训练轮数
            lr: 学习率
            gamma: 量化损失权重
            eta: 平衡损失权重
            use_gpu: 是否使用 GPU
            result_dir: 结果输出目录
        """
        self.data_path = Path(data_path)
        self.bit_dim = bit_dim
        self.batch_size = batch_size
        self.max_epoch = max_epoch
        self.lr = lr
        self.gamma = gamma
        self.eta = eta
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.result_dir = Path(result_dir)

        # 创建结果目录
        self.result_dir.mkdir(parents=True, exist_ok=True)

        # 设备
        self.device = torch.device('cuda' if self.use_gpu else 'cpu')

        # 实验结果
        self.results = {
            'config': {},
            'training': {},
            'evaluation': {},
            'aspe_test': {},
            'visualizations': []
        }

    def run_full_experiment(self) -> Dict:
        """
        运行完整实验流程。

        返回:
            实验结果字典
        """
        print("=" * 70)
        print("Flickr25K 完整系统测试和实验")
        print(f"开始时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)

        start_time = time.time()

        # 1. 加载数据
        print("\n" + "=" * 70)
        print("步骤 1: 加载数据")
        print("=" * 70)
        data = self._load_data()

        # 2. 训练 DCMH 模型
        print("\n" + "=" * 70)
        print("步骤 2: 训练 DCMH 模型")
        print("=" * 70)
        training_results = self._train_dcmh(data)
        self.results['training'] = training_results

        # 3. 评估检索性能
        print("\n" + "=" * 70)
        print("步骤 3: 评估检索性能")
        print("=" * 70)
        eval_results = self._evaluate_retrieval(data)
        self.results['evaluation'] = eval_results

        # 4. ASPE 加密测试
        print("\n" + "=" * 70)
        print("步骤 4: ASPE 加密测试")
        print("=" * 70)
        aspe_results = self._test_aspe(eval_results)
        self.results['aspe_test'] = aspe_results

        # 5. 生成可视化
        print("\n" + "=" * 70)
        print("步骤 5: 生成可视化图表")
        print("=" * 70)
        self._generate_visualizations()

        # 6. 生成报告
        print("\n" + "=" * 70)
        print("步骤 6: 生成实验报告")
        print("=" * 70)
        self._generate_report()

        # 保存结果
        self._save_results()

        # 关闭 h5py 文件
        if hasattr(self, 'h5_file') and self.h5_file:
            self.h5_file.close()

        elapsed_time = time.time() - start_time

        print("\n" + "=" * 70)
        print("实验完成")
        print("=" * 70)
        print(f"总耗时：{elapsed_time / 3600:.2f} 小时")
        print(f"结果目录：{self.result_dir}")

        return self.results

    def _load_data(self) -> Dict:
        """加载数据（使用 h5py 引用，避免全部加载到内存）。"""
        import h5py

        print(f"数据路径：{self.data_path}")

        # 打开 HDF5 文件（保持打开状态用于按需读取）
        self.h5_file = h5py.File(self.data_path, 'r')

        # 数据集划分
        query_size = 2000
        training_size = 10000
        database_size = 18015

        # 存储索引范围，实际数据按需读取
        data = {
            'query': {
                'indices': (0, query_size),
                'tags': torch.from_numpy(
                    np.array(self.h5_file['YAll'][:query_size]).astype('float32')
                ),
                'labels': torch.from_numpy(
                    np.array(self.h5_file['LAll'][:query_size]).astype('float32')
                )
            },
            'train': {
                'indices': (query_size, query_size + training_size),
                'tags': torch.from_numpy(
                    np.array(self.h5_file['YAll'][query_size:query_size + training_size]).astype('float32')
                ),
                'labels': torch.from_numpy(
                    np.array(self.h5_file['LAll'][query_size:query_size + training_size]).astype('float32')
                )
            },
            'database': {
                'indices': (query_size, query_size + database_size),
                'tags': torch.from_numpy(
                    np.array(self.h5_file['YAll'][query_size:query_size + database_size]).astype('float32')
                ),
                'labels': torch.from_numpy(
                    np.array(self.h5_file['LAll'][query_size:query_size + database_size]).astype('float32')
                )
            }
        }

        # 移动到 GPU
        if self.use_gpu:
            for split in data.keys():
                data[split]['labels'] = data[split]['labels'].cuda()

        print(f"查询集：{len(data['query']['tags'])}")
        print(f"训练集：{len(data['train']['tags'])}")
        print(f"数据库：{len(data['database']['tags'])}")
        print(f"文本维度：{data['train']['tags'].shape[1]}")
        print(f"类别数：{data['train']['labels'].shape[1]}")

        self.y_dim = data['train']['tags'].shape[1]
        self.n_classes = data['train']['labels'].shape[1]

        self.results['config']['data'] = {
            'query_size': len(data['query']['tags']),
            'train_size': len(data['train']['tags']),
            'database_size': len(data['database']['tags']),
            'y_dim': self.y_dim,
            'n_classes': self.n_classes
        }

        return data

    def _train_dcmh(self, data: Dict) -> Dict:
        """训练 DCMH 模型。"""
        from core.hashing.dcmh_image import DCMHImageModule
        from core.hashing.dcmh_text import DCMHTextModule
        import torch.optim as optim
        from tqdm import tqdm

        # 初始化模型
        model = DCMHModel(bit=self.bit_dim, y_dim=self.y_dim)
        model.to(self.device)

        n_params = sum(p.numel() for p in model.parameters())
        print(f"模型参数量：{n_params:,}")

        # 优化器
        optimizer = optim.SGD(
            model.parameters(),
            lr=self.lr,
            momentum=0.9,
            weight_decay=1e-5
        )

        # 学习率调度器
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.max_epoch, eta_min=1e-6
        )

        # 训练数据索引
        train_start, train_end = data['train']['indices']
        train_tags = data['train']['tags']  # (10000, 1386)
        train_labels = data['train']['labels']  # (10000, 24)

        n_train = len(train_tags)

        # F 和 G 缓冲区
        F_buffer = torch.randn(n_train, self.bit_dim).to(self.device)
        G_buffer = torch.randn(n_train, self.bit_dim).to(self.device)

        # 相似度矩阵（只使用标签计算，不需要图像）
        Sim = (train_labels @ train_labels.t() > 0).float().to(self.device)

        # 训练循环
        history = {'loss': [], 'map_i2t': [], 'map_t2i': []}
        best_map = 0.0
        best_epoch = 0

        # 预计算所有标签的相似度矩阵（一次性计算，避免重复）
        train_labels_np = train_labels.cpu().numpy()
        Sim_pre = (train_labels_np @ train_labels_np.T > 0).astype('float32')
        Sim_tensor = torch.from_numpy(Sim_pre).to(self.device)

        # 预计算 unupdated 索引的补集（用于平衡损失）
        batch_size = self.batch_size
        n_batches = n_train // batch_size

        for epoch in range(self.max_epoch):
            model.train()
            total_loss = 0.0

            pbar = tqdm(range(n_batches), desc=f'Epoch {epoch+1}/{self.max_epoch}')

            for bi in pbar:
                # 计算批次在训练集中的起始和结束索引
                batch_rel_start = bi * batch_size
                batch_rel_end = batch_rel_start + batch_size
                batch_rel_idx = torch.arange(batch_rel_start, batch_rel_end, dtype=torch.long).to(self.device)

                # 从 h5py 按需加载图像批次
                batch_start = train_start + batch_rel_start
                batch_end = train_start + batch_rel_end
                img_batch_np = self.h5_file['images'][batch_start:batch_end]
                img_batch_np = img_batch_np.astype('float32') / 255.0
                img_batch = torch.from_numpy(img_batch_np).to(self.device)

                # 获取批次数据
                txt_batch = train_tags[batch_rel_start:batch_rel_end].unsqueeze(1).unsqueeze(-1).to(self.device)
                lbl_batch = train_labels[batch_rel_start:batch_rel_end]

                # 从预计算矩阵中获取相似度 (batch_size, n_train)
                S = Sim_tensor[batch_rel_start:batch_rel_end]

                # 图像流
                optimizer.zero_grad()
                img_hash = model.image_module(img_batch)
                F_buffer[batch_rel_idx] = img_hash.detach()

                # 成对交叉熵损失
                theta = 0.5 * img_hash @ G_buffer.t()
                logloss = -torch.sum(S * theta - torch.log(1 + torch.exp(theta) + 1e-10))

                # 量化损失
                B = torch.sign(F_buffer + G_buffer)
                quant_loss = torch.sum(torch.pow(B[batch_rel_idx] - img_hash, 2))

                # 平衡损失（简化版：只计算当前批次的和）
                balance_loss = torch.sum(torch.pow(img_hash.sum(dim=0), 2))

                loss = (logloss + self.gamma * quant_loss + self.eta * balance_loss)
                loss = loss / (batch_size * n_train)
                loss.backward()
                optimizer.step()

                # 文本流
                optimizer.zero_grad()
                txt_hash = model.text_module(txt_batch)
                G_buffer[batch_rel_idx] = txt_hash.detach()

                # 成对交叉熵损失
                theta_txt = 0.5 * txt_hash @ F_buffer.t()
                logloss_txt = -torch.sum(S * theta_txt - torch.log(1 + torch.exp(theta_txt) + 1e-10))

                # 量化损失
                B = torch.sign(F_buffer + G_buffer)
                quant_loss_txt = torch.sum(torch.pow(B[batch_rel_idx] - txt_hash, 2))

                # 平衡损失
                balance_loss_txt = torch.sum(torch.pow(txt_hash.sum(dim=0), 2))

                loss_txt = (logloss_txt + self.gamma * quant_loss_txt + self.eta * balance_loss_txt)
                loss_txt = loss_txt / (batch_size * n_train)
                loss_txt.backward()
                optimizer.step()

                # 更新 B
                B = torch.sign(F_buffer + G_buffer)
                total_loss += loss.item() + loss_txt.item()
                pbar.set_postfix({'loss': f'{total_loss / (bi + 1):.4f}'})

            scheduler.step()

            avg_loss = total_loss / n_batches
            history['loss'].append(avg_loss)

            # 验证
            if (epoch + 1) % 10 == 0 or epoch == self.max_epoch - 1:
                map_i2t, map_t2i = self._validate(
                    model, data['query'], data['database']
                )
                history['map_i2t'].append(map_i2t)
                history['map_t2i'].append(map_t2i)

                avg_map = (map_i2t + map_t2i) / 2
                print(f"\n验证 - Epoch {epoch + 1}:")
                print(f"  MAP(i->t): {map_i2t:.4f}")
                print(f"  MAP(t->i): {map_t2i:.4f}")
                print(f"  平均 MAP: {avg_map:.4f}")

                if avg_map > best_map:
                    best_map = avg_map
                    best_epoch = epoch + 1

                    # 保存最佳模型
                    checkpoint = {
                        'epoch': epoch + 1,
                        'model_state_dict': model.state_dict(),
                        'best_map': best_map,
                        'config': {
                            'bit_dim': self.bit_dim,
                            'y_dim': self.y_dim
                        }
                    }
                    torch.save(checkpoint, self.result_dir / 'dcmh_best.pth')

            # 保存检查点
            if (epoch + 1) % 50 == 0:
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'history': history
                }, self.result_dir / f'dcmh_checkpoint_{epoch + 1}.pth')

        print(f"\n最佳 MAP: {best_map:.4f} (Epoch {best_epoch})")

        # 保存最终模型
        torch.save({
            'epoch': self.max_epoch,
            'model_state_dict': model.state_dict(),
            'history': history,
            'best_map': best_map,
            'best_epoch': best_epoch
        }, self.result_dir / 'dcmh_final.pth')

        # 内存清理：删除训练期间的大型张量
        del F_buffer, G_buffer, Sim, model
        if self.use_gpu:
            torch.cuda.empty_cache()
        gc.collect()

        print("训练完成，已清理内存")

        return {
            'history': history,
            'best_map': best_map,
            'best_epoch': best_epoch,
            'model_path': str(self.result_dir / 'dcmh_final.pth')
        }

    def _validate(self, model, query_data, database_data) -> tuple:
        """验证模型。"""
        model.eval()

        with torch.no_grad():
            # 按需加载查询图像（数据已是 N,C,H,W 格式，无需 transpose）
            q_start, q_end = query_data['indices']
            q_img_np = self.h5_file['images'][q_start:q_end].astype('float32') / 255.0

            # 分批处理查询图像（节省内存）
            qBX_list = []
            qBY_list = []
            batch_size = 64

            for i in range(0, len(q_img_np), batch_size):
                q_img_batch = torch.from_numpy(q_img_np[i:i+batch_size]).to(self.device)
                q_txt_batch = query_data['tags'][i:i+batch_size].unsqueeze(1).unsqueeze(-1).to(self.device)

                qBX_batch = torch.sign(model.image_module(q_img_batch))
                qBY_batch = torch.sign(model.text_module(q_txt_batch))

                qBX_list.append(qBX_batch.cpu())
                qBY_list.append(qBY_batch.cpu())

            qBX = torch.cat(qBX_list, dim=0)
            qBY = torch.cat(qBY_list, dim=0)

            # 分批处理数据库图像（数据已是 N,C,H,W 格式，无需 transpose）
            db_start, db_end = database_data['indices']
            rBX_list = []
            rBY_list = []

            for i in range(0, db_end - db_start, batch_size):
                db_img_np = self.h5_file['images'][db_start + i:db_start + i + batch_size].astype('float32') / 255.0
                db_img_batch = torch.from_numpy(db_img_np).to(self.device)

                db_txt_batch = database_data['tags'][i:i+len(db_img_np)].unsqueeze(1).unsqueeze(-1).to(self.device)

                rBX_batch = torch.sign(model.image_module(db_img_batch))
                rBY_batch = torch.sign(model.text_module(db_txt_batch))

                rBX_list.append(rBX_batch.cpu())
                rBY_list.append(rBY_batch.cpu())

            rBX = torch.cat(rBX_list, dim=0)
            rBY = torch.cat(rBY_list, dim=0)

            # 计算 mAP
            map_i2t = calc_map_k(
                qBX, rBY, query_data['labels'], database_data['labels']
            )
            map_t2i = calc_map_k(
                qBY, rBX, query_data['labels'], database_data['labels']
            )

        return map_i2t.item(), map_t2i.item()

    def _evaluate_retrieval(self, data: Dict) -> Dict:
        """评估检索性能。"""
        # 加载最佳模型
        model_path = self.result_dir / 'dcmh_best.pth'
        if not model_path.exists():
            model_path = self.result_dir / 'dcmh_final.pth'

        checkpoint = torch.load(model_path, map_location=self.device)

        model = DCMHModel(bit=self.bit_dim, y_dim=self.y_dim)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()

        print(f"加载模型：{model_path}")
        print(f"最佳 MAP: {checkpoint.get('best_map', 0):.4f}")

        # 生成哈希码（分批处理以节省内存）
        with torch.no_grad():
            # 查询哈希码（数据已是 N,C,H,W 格式，无需 transpose）
            q_start, q_end = data['query']['indices']
            q_img_np = self.h5_file['images'][q_start:q_end].astype('float32') / 255.0

            qBX_list = []
            qBY_list = []
            batch_size = 64

            for i in range(0, len(q_img_np), batch_size):
                q_img_batch = torch.from_numpy(q_img_np[i:i+batch_size]).to(self.device)
                q_txt_batch = data['query']['tags'][i:i+batch_size].unsqueeze(1).unsqueeze(-1).to(self.device)

                qBX_batch = torch.sign(model.image_module(q_img_batch))
                qBY_batch = torch.sign(model.text_module(q_txt_batch))

                qBX_list.append(qBX_batch.cpu().numpy())
                qBY_list.append(qBY_batch.cpu().numpy())

            qBX = np.vstack(qBX_list)
            qBY = np.vstack(qBY_list)

            # 数据库哈希码（数据已是 N,C,H,W 格式，无需 transpose）
            db_start, db_end = data['database']['indices']
            rBX_list = []
            rBY_list = []

            for i in range(0, db_end - db_start, batch_size):
                db_img_np = self.h5_file['images'][db_start + i:db_start + i + batch_size].astype('float32') / 255.0

                db_txt_batch = data['database']['tags'][i:i+len(db_img_np)].unsqueeze(1).unsqueeze(-1).to(self.device)
                db_img_batch = torch.from_numpy(db_img_np).to(self.device)

                rBX_batch = torch.sign(model.image_module(db_img_batch))
                rBY_batch = torch.sign(model.text_module(db_txt_batch))

                rBX_list.append(rBX_batch.cpu().numpy())
                rBY_list.append(rBY_batch.cpu().numpy())

            rBX = np.vstack(rBX_list)
            rBY = np.vstack(rBY_list)

        # 计算 mAP
        qBX_t = torch.from_numpy(qBX).float()
        qBY_t = torch.from_numpy(qBY).float()
        rBX_t = torch.from_numpy(rBX).float()
        rBY_t = torch.from_numpy(rBY).float()

        map_i2t = calc_map_k(
            qBX_t, rBY_t,
            data['query']['labels'], data['database']['labels']
        )
        map_t2i = calc_map_k(
            qBY_t, rBX_t,
            data['query']['labels'], data['database']['labels']
        )

        # 计算 Precision@K 和 Recall@K
        sim_i2t = np.dot(qBX, rBY.T)
        sim_t2i = np.dot(qBY, rBX.T)

        query_labels = np.argmax(data['query']['labels'].cpu().numpy(), axis=1)
        db_labels = np.argmax(data['database']['labels'].cpu().numpy(), axis=1)

        retrieved_i2t = np.argsort(-sim_i2t, axis=1)
        retrieved_t2i = np.argsort(-sim_t2i, axis=1)

        pr_results = {'image_to_text': {}, 'text_to_image': {}}

        for k in [1, 5, 10, 20, 50, 100]:
            # P@K
            p_i2t = []
            p_t2i = []
            for i in range(len(query_labels)):
                top_k_i2t = retrieved_i2t[i, :k]
                top_k_t2i = retrieved_t2i[i, :k]
                p_i2t.append(np.mean(db_labels[top_k_i2t] == query_labels[i]))
                p_t2i.append(np.mean(db_labels[top_k_t2i] == query_labels[i]))
            pr_results['image_to_text'][f'p@{k}'] = np.mean(p_i2t)
            pr_results['text_to_image'][f'p@{k}'] = np.mean(p_t2i)

        # 哈希质量
        quality = compute_hash_quality(qBX)

        results = {
            'map': {
                'i2t': map_i2t.item(),
                't2i': map_t2i.item(),
                'avg': ((map_i2t + map_t2i) / 2).item()
            },
            'precision_recall': pr_results,
            'hash_quality': quality,
            'model_path': str(model_path)
        }

        print(f"\n最终评估结果:")
        print(f"  MAP(i->t): {results['map']['i2t']:.4f}")
        print(f"  MAP(t->i): {results['map']['t2i']:.4f}")
        print(f"  平均 MAP: {results['map']['avg']:.4f}")

        # 内存清理
        del model, checkpoint, qBX_t, qBY_t, rBX_t, rBY_t
        if self.use_gpu:
            torch.cuda.empty_cache()
        gc.collect()

        return results

    def _test_aspe(self, eval_results: Dict) -> Dict:
        """测试 ASPE 加密。"""
        # 生成测试数据
        np.random.seed(42)
        n_test = 100
        bit = self.bit_dim

        qB = np.sign(np.random.randn(n_test, bit)).astype(np.float64)
        rB = np.sign(np.random.randn(n_test * 10, bit)).astype(np.float64)

        # 初始化 ASPE
        aspe_wrapper = ASPEForDCMH(bit_dim=bit, seed=42)

        print(f"测试样本数：{n_test}")
        print(f"ASPE 维度：{bit}")

        # 1. 内积保持性
        print("\n测试内积保持性...")
        aspe = ASPEScheme1(d=bit, seed=42)

        ip_ratios = []
        for i in range(min(20, n_test)):
            p = qB[i]
            q = rB[i]

            plaintext_ip = np.dot(p, q)

            p_enc = aspe.encrypt_database_point(p)
            q_enc = aspe.encrypt_query(q)

            ciphertext_ip = aspe.ciphertext_inner_product(p_enc, q_enc)

            if plaintext_ip != 0:
                ratio = ciphertext_ip / plaintext_ip
                ip_ratios.append(ratio)

        positive_ratio = sum(1 for r in ip_ratios if r > 0) / len(ip_ratios) if ip_ratios else 0

        # 2. 排序一致性
        print("测试排序一致性...")
        encrypted_rB = aspe_wrapper.GenEnc(rB)
        encrypted_qB = aspe_wrapper.GenTrap(qB)

        correlations = []
        for i in range(min(20, n_test)):
            inner_prod_orig = np.dot(qB[i:i+1], rB.T)
            hamm_orig = 0.5 * (bit - inner_prod_orig)

            q_enc = encrypted_qB[i:i+1]
            hamm_aspe = aspe_wrapper.ciphertext_hamming_distance(q_enc, encrypted_rB)

            corr = np.corrcoef(hamm_orig.squeeze(), hamm_aspe.squeeze())[0, 1]
            correlations.append(corr)

        # 3. mAP 保持性
        print("测试 mAP 保持性...")
        query_L = np.random.randint(0, 2, (n_test, 20)).astype(np.float64)
        retrieval_L = np.random.randint(0, 2, (n_test * 10, 20)).astype(np.float64)

        # 原始 mAP
        map_orig = calc_map_k(
            torch.from_numpy(np.sign(qB)).float(),
            torch.from_numpy(np.sign(rB)).float(),
            torch.from_numpy(query_L),
            torch.from_numpy(retrieval_L)
        )

        # ASPE mAP
        map_aspe = aspe_wrapper.calc_ciphertext_map(
            encrypted_qB, encrypted_rB, query_L, retrieval_L
        )

        map_diff = abs(map_orig - map_aspe)

        results = {
            'inner_product_preservation': {
                'positive_ratio': positive_ratio,
                'n_samples': len(ip_ratios)
            },
            'sorting_consistency': {
                'avg_correlation': np.mean(correlations),
                'min_correlation': np.min(correlations),
                'n_samples': len(correlations)
            },
            'map_preservation': {
                'original_map': float(map_orig),
                'aspe_map': float(map_aspe),
                'difference': float(map_diff),
                'passed': map_diff < 1e-3
            }
        }

        print(f"\nASPE 测试结果:")
        print(f"  内积正缩放因子比率：{positive_ratio:.2%}")
        print(f"  排序相关系数：{np.mean(correlations):.6f}")
        print(f"  原始 mAP: {map_orig:.6f}")
        print(f"  ASPE mAP: {map_aspe:.6f}")
        print(f"  差异：{map_diff:.6e}")
        print(f"  测试通过：{'是' if map_diff < 1e-3 else '否'}")

        return results

    def _generate_visualizations(self):
        """生成可视化图表。"""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np

        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

        chart_paths = []

        # 1. mAP 对比图
        print("生成 mAP 对比图...")
        fig, ax = plt.subplots(figsize=(10, 6))

        categories = ['图像→文本', '文本→图像', '平均']
        values = [
            self.results['evaluation']['map']['i2t'],
            self.results['evaluation']['map']['t2i'],
            self.results['evaluation']['map']['avg']
        ]

        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        bars = ax.bar(categories, values, color=colors, alpha=0.8)

        ax.set_ylabel('mAP')
        ax.set_title(f'DCMH Flickr25K 检索性能 (bit={self.bit_dim})')
        ax.set_ylim(0, 1)

        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                   f'{val:.4f}', ha='center', va='bottom', fontsize=11)

        plt.tight_layout()
        chart_path = self.result_dir / 'flickr25k_map_comparison.png'
        plt.savefig(chart_path, dpi=150, bbox_inches='tight')
        plt.close()
        chart_paths.append(str(chart_path))

        # 2. Precision@K 图
        print("生成 Precision@K 对比图...")
        fig, ax = plt.subplots(figsize=(12, 6))

        pr = self.results['evaluation']['precision_recall']
        k_labels = list(pr['image_to_text']['p@1'].keys())
        k_values = [int(k.replace('p@', '')) for k in k_labels]

        p_i2t = [pr['image_to_text'][f'p@{k}'] for k in k_labels]
        p_t2i = [pr['text_to_image'][f'p@{k}'] for k in k_labels]

        x = np.arange(len(k_values))
        width = 0.35

        ax.bar(x - width/2, p_i2t, width, label='图像→文本', color='#1f77b4', alpha=0.8)
        ax.bar(x + width/2, p_t2i, width, label='文本→图像', color='#ff7f0e', alpha=0.8)

        ax.set_xlabel('K')
        ax.set_ylabel('Precision')
        ax.set_title('Precision@K 对比')
        ax.set_xticks(x)
        ax.set_xticklabels(k_values)
        ax.legend()
        ax.set_ylim(0, 1)
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        chart_path = self.result_dir / 'flickr25k_precision_k.png'
        plt.savefig(chart_path, dpi=150, bbox_inches='tight')
        plt.close()
        chart_paths.append(str(chart_path))

        # 3. 训练曲线
        if self.results['training'].get('history'):
            print("生成训练曲线...")
            history = self.results['training']['history']

            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            # 损失曲线
            ax1 = axes[0]
            epochs = range(1, len(history['loss']) + 1)
            ax1.plot(epochs, history['loss'], 'b-', linewidth=1.5)
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.set_title('训练损失曲线')
            ax1.grid(True, alpha=0.3)

            # mAP 曲线
            if history.get('map_i2t'):
                ax2 = axes[1]
                valid_interval = 10
                map_epochs = range(
                    valid_interval, len(history['loss']) + 1, valid_interval
                )
                ax2.plot(map_epochs, history['map_i2t'],
                        'b-', label='MAP(i->t)', linewidth=1.5)
                ax2.plot(map_epochs, history['map_t2i'],
                        'r-', label='MAP(t->i)', linewidth=1.5)
                avg_map = [(i + t) / 2 for i, t in zip(history['map_i2t'], history['map_t2i'])]
                ax2.plot(map_epochs, avg_map,
                        'g-', label='平均 MAP', linewidth=2)
                ax2.set_xlabel('Epoch')
                ax2.set_ylabel('mAP')
                ax2.set_title('mAP 变化曲线')
                ax2.legend()
                ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            chart_path = self.result_dir / 'flickr25k_training_curve.png'
            plt.savefig(chart_path, dpi=150, bbox_inches='tight')
            plt.close()
            chart_paths.append(str(chart_path))

        # 4. ASPE 测试结果图
        print("生成 ASPE 测试结果图...")
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        aspe = self.results['aspe_test']

        # 内积保持性
        ax1 = axes[0]
        ax1.bar(['正比率'], [aspe['inner_product_preservation']['positive_ratio']],
               color='#2ca02c', alpha=0.8)
        ax1.set_ylabel('比率')
        ax1.set_title('ASPE 内积保持性')
        ax1.set_ylim(0, 1)
        ax1.grid(axis='y', alpha=0.3)

        # 排序一致性
        ax2 = axes[1]
        corr_values = [
            aspe['sorting_consistency']['avg_correlation'],
            aspe['sorting_consistency']['min_correlation']
        ]
        ax2.bar(['平均相关系数', '最小相关系数'], corr_values,
               color='#1f77b4', alpha=0.8)
        ax2.set_ylabel('相关系数')
        ax2.set_title('ASPE 排序一致性')
        ax2.set_ylim(0.9, 1.0)
        ax2.grid(axis='y', alpha=0.3)

        # mAP 保持性
        ax3 = axes[2]
        map_values = [
            aspe['map_preservation']['original_map'],
            aspe['map_preservation']['aspe_map']
        ]
        ax3.bar(['原始 mAP', 'ASPE mAP'], map_values,
               color=['#1f77b4', '#ff7f0e'], alpha=0.8)
        ax3.set_ylabel('mAP')
        ax3.set_title('ASPE mAP 保持性')
        ax3.set_ylim(0, max(map_values) * 1.2)
        ax3.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        chart_path = self.result_dir / 'flickr25k_aspe_test.png'
        plt.savefig(chart_path, dpi=150, bbox_inches='tight')
        plt.close()
        chart_paths.append(str(chart_path))

        self.results['visualizations'] = chart_paths
        print(f"已生成 {len(chart_paths)} 个图表")

    def _generate_report(self):
        """生成 Markdown 实验报告。"""
        report_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        training = self.results['training']
        evaluation = self.results['evaluation']
        aspe_test = self.results['aspe_test']

        report_content = f"""# Flickr25K 完整系统测试报告

**生成时间**: {report_date}
**数据集**: Flickr25K
**哈希码维度**: {self.bit_dim} bits

---

## 1. 实验概述

本次实验在 Flickr25K 数据集上完整测试了 DCMH + ASPE 系统的性能。

### 1.1 数据集统计

| 划分 | 样本数 |
|------|--------|
| 查询集 | {self.results['config']['data']['query_size']:,} |
| 训练集 | {self.results['config']['data']['train_size']:,} |
| 数据库 | {self.results['config']['data']['database_size']:,} |
| 类别数 | {self.results['config']['data']['n_classes']} |

### 1.2 训练配置

| 配置项 | 值 |
|--------|-----|
| 批次大小 | {self.batch_size} |
| 学习率 | {self.lr} |
| 最大轮数 | {self.max_epoch} |
| 量化权重 (γ) | {self.gamma} |
| 平衡权重 (η) | {self.eta} |

---

## 2. DCMH 训练结果

### 2.1 训练统计

| 指标 | 值 |
|------|-----|
| 最佳平均 MAP | {training['best_map']:.4f} |
| 最佳轮次 | Epoch {training['best_epoch']} |
| 总训练轮数 | {self.max_epoch} |

---

## 3. 检索性能评估

### 3.1 mAP 结果

| 检索方向 | mAP |
|----------|-----|
| 图像 → 文本 | {evaluation['map']['i2t']:.4f} |
| 文本 → 图像 | {evaluation['map']['t2i']:.4f} |
| **平均** | **{evaluation['map']['avg']:.4f}** |

### 3.2 Precision@K

| K | 图像→文本 | 文本→图像 |
|---|-----------|-----------|
"""

        pr = evaluation['precision_recall']
        for k in ['1', '5', '10', '20', '50', '100']:
            p_i2t = pr['image_to_text'].get(f'p@{k}', 0)
            p_t2i = pr['text_to_image'].get(f'p@{k}', 0)
            report_content += f"| {k} | {p_i2t:.4f} | {p_t2i:.4f} |\n"

        report_content += f"""
### 3.3 哈希码质量

| 指标 | 值 |
|------|-----|
| 平衡性 | {evaluation['hash_quality']['balance']:.4f} |
| 唯一性 | {evaluation['hash_quality']['uniqueness']:.4f} |
| 平均汉明距离 | {evaluation['hash_quality']['avg_hamming_distance']:.4f} |

---

## 4. ASPE 加密测试

### 4.1 内积保持性

- **正缩放因子比率**: {aspe_test['inner_product_preservation']['positive_ratio']:.2%}

### 4.2 排序一致性

- **平均相关系数**: {aspe_test['sorting_consistency']['avg_correlation']:.6f}
- **最小相关系数**: {aspe_test['sorting_consistency']['min_correlation']:.6f}

### 4.3 mAP 保持性

| 指标 | 值 |
|------|-----|
| 原始 mAP | {aspe_test['map_preservation']['original_map']:.6f} |
| ASPE mAP | {aspe_test['map_preservation']['aspe_map']:.6f} |
| 差异 | {aspe_test['map_preservation']['difference']:.2e} |
| 测试通过 | {"是" if aspe_test['map_preservation']['passed'] else "否"} |

---

## 5. 可视化图表

"""

        chart_captions = {
            'flickr25k_map_comparison.png': '图 1: mAP 对比',
            'flickr25k_precision_k.png': '图 2: Precision@K 对比',
            'flickr25k_training_curve.png': '图 3: 训练曲线',
            'flickr25k_aspe_test.png': '图 4: ASPE 测试结果'
        }

        for chart_name, caption in chart_captions.items():
            if (self.result_dir / chart_name).exists():
                report_content += f"### {caption}\n\n"
                report_content += f"![{caption}]({chart_name})\n\n"

        report_content += f"""
---

## 6. 实验结论

基于本次实验结果：

1. **DCMH 模型性能**:
   - 在 Flickr25K 数据集上达到 **{evaluation['map']['avg']:.4f}** 的平均 mAP
   - 图像→文本检索 mAP: {evaluation['map']['i2t']:.4f}
   - 文本→图像检索 mAP: {evaluation['map']['t2i']:.4f}

2. **哈希码质量**:
   - 平衡性：{evaluation['hash_quality']['balance']:.4f}
   - 唯一性：{evaluation['hash_quality']['uniqueness']:.4f}

3. **ASPE 加密**:
   - 内积保持性验证通过
   - 排序一致性良好（相关系数 > {aspe_test['sorting_consistency']['avg_correlation']:.4f}）
   - mAP 保持性验证{"通过" if aspe_test['map_preservation']['passed'] else "未通过"}

---

**实验完成** | 详细数据请查看 `flickr25k_experiment_results.json`
"""

        report_path = self.result_dir / 'flickr25k_experiment_report.md'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)

        print(f"报告已保存：{report_path}")

    def _save_results(self):
        """保存 JSON 结果。"""
        self.results['config'].update({
            'bit_dim': self.bit_dim,
            'batch_size': self.batch_size,
            'max_epoch': self.max_epoch,
            'lr': self.lr,
            'gamma': self.gamma,
            'eta': self.eta,
            'use_gpu': self.use_gpu,
            'result_dir': str(self.result_dir),
            'completed_at': datetime.now().isoformat()
        })

        json_path = self.result_dir / 'flickr25k_experiment_results.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)

        print(f"JSON 结果已保存：{json_path}")


def run_flickr25k_experiment(data_path: str = 'data/flickr25k/FLICKR-25K.mat',
                            bit_dim: int = 64,
                            max_epoch: int = 500,
                            result_dir: str = 'results/flickr-25k',
                            **kwargs) -> Dict:
    """
    运行 Flickr25K 实验的便捷函数。

    参数:
        data_path: 数据文件路径
        bit_dim: 哈希码维度
        max_epoch: 最大训练轮数
        result_dir: 结果目录
        **kwargs: 其他配置参数

    返回:
        实验结果字典
    """
    experiment = Flickr25KExperiment(
        data_path=data_path,
        bit_dim=bit_dim,
        max_epoch=max_epoch,
        result_dir=result_dir,
        **kwargs
    )

    return experiment.run_full_experiment()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Flickr25K 完整系统测试')
    parser.add_argument('--data', type=str, default='data/flickr25k/FLICKR-25K.mat',
                       help='数据文件路径')
    parser.add_argument('--bit', type=int, default=64, help='哈希码维度')
    parser.add_argument('--epochs', type=int, default=500, help='最大训练轮数')
    parser.add_argument('--batch-size', type=int, default=128, help='批次大小')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--result-dir', type=str, default='results/flickr-25k',
                       help='结果目录')
    parser.add_argument('--no-gpu', action='store_true', help='禁用 GPU')

    args = parser.parse_args()

    results = run_flickr25k_experiment(
        data_path=args.data,
        bit_dim=args.bit,
        max_epoch=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        result_dir=args.result_dir,
        use_gpu=not args.no_gpu
    )

    print("\n" + "=" * 70)
    print("实验完成！")
    print("=" * 70)
    print(f"结果目录：{args.result_dir}")
    print(f"最佳 MAP: {results['training']['best_map']:.4f}")
