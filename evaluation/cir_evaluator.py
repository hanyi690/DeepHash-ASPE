"""
CNN 图像检索 (CIR) 评估器

提供 CNN 图像检索系统的完整评估功能：
- 使用 GL18/ResNet101-GeM 模型提取特征
- 计算 mAP、Precision@K、Recall@K
- 支持 Oxford5k/Paris6k 数据集
- ASPE 加密验证（密文 mAP 与明文 mAP 一致性）
- 生成可视化图表和评估报告
- 支持特征缓存（避免重复提取）
"""

import os
import sys
import json
import math
import numpy as np
import torch
import torchvision.transforms as transforms
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Union
from PIL import Image
import logging

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.cirtorch.networks.imageretrievalnet import init_network, extract_vectors
from core.cirtorch.datasets.testdataset import configdataset
from core.cirtorch.datasets.genericdataset import ImagesFromList
from core.cirtorch.utils.evaluate import compute_map, compute_ap
from core.aspe.cnn_wrapper import ASPEForCNN
from backend.app.services.hash_cache_service import HashCacheService, CIR_CACHE_DIR
from evaluation.visualization import (
    plot_map_comparison,
    plot_cir_map_comparison,
    plot_precision_recall,
    plot_aspe_comparison,
    plot_cir_aspe_comparison,
    generate_evaluation_report
)

logger = logging.getLogger(__name__)


def _clean_nans(obj):
    """
    递归清理 NaN 和 Inf，将其转换为 None。

    处理嵌套的 list、dict、numpy 数组等。
    """
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, np.floating):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return _clean_nans(obj.tolist())
    if isinstance(obj, list):
        return [_clean_nans(item) for item in obj]
    if isinstance(obj, dict):
        return {k: _clean_nans(v) for k, v in obj.items()}
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    return obj


class NaNAwareJSONEncoder(json.JSONEncoder):
    """
    自定义 JSON 编码器，将 NaN/Inf 转换为 null。
    解决 JSON 标准不支持 NaN 的问题。
    """
    def default(self, obj):
        if isinstance(obj, float):
            if math.isnan(obj) or math.isinf(obj):
                return None
        # 处理 numpy 类型
        if isinstance(obj, np.floating):
            if np.isnan(obj) or np.isinf(obj):
                return None
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        # 处理 torch 类型
        if hasattr(obj, 'item'):  # torch.Tensor 标量
            return obj.item()
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        return super().default(obj)


class CIREvaluator:
    """
    CNN 图像检索评估器。

    支持标准图像检索评估（Oxford5k/Paris6k 数据集），
    以及 ASPE 加密验证。

    使用示例：
        # 基本用法
        evaluator = CIREvaluator(
            model_path='data/networks/gl18-tl-resnet101-gem-w-a4d43db.pth',
            data_dir='data/roxford5k',
            dataset='roxford5k'
        )
        results = evaluator.evaluate()

        # 包含 ASPE 验证
        results = evaluator.evaluate(run_aspe=True)
    """

    # 支持的数据集
    DATASETS = ['oxford5k', 'paris6k', 'roxford5k', 'rparis6k']

    def __init__(self,
                 model_path: str,
                 data_dir: str,
                 dataset: str = 'roxford5k',
                 use_gpu: bool = True,
                 image_size: int = 1024,
                 multiscale: List[int] = None):
        """
        初始化 CIR 评估器。

        参数：
            model_path: 预训练模型路径 (.pth 文件)
            data_dir: 数据集根目录
            dataset: 数据集名称 ('oxford5k', 'paris6k', 'roxford5k', 'rparis6k')
            use_gpu: 是否使用 GPU
            image_size: 图像缩放大小
            multiscale: 多尺度列表（默认 [1] 单尺度）
        """
        self.model_path = Path(model_path)
        self.data_dir = Path(data_dir)
        self.dataset = dataset.lower()
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = torch.device('cuda' if self.use_gpu else 'cpu')
        self.image_size = image_size
        self.multiscale = multiscale if multiscale else [1]

        # 验证数据集
        if self.dataset not in self.DATASETS:
            raise ValueError(f"不支持的数据集: {dataset}，可选: {self.DATASETS}")

        # 加载模型和数据配置
        self.model = None
        self.meta = None
        self.cfg = None

        self._load_model()
        self._load_dataset_config()

        logger.info(f"CIREvaluator 初始化完成: dataset={dataset}, device={self.device}")

    def _load_model(self):
        """
        加载 CNN 图像检索模型。

        使用 core.cirtorch.networks.imageretrievalnet 模块。
        """
        print("=" * 60)
        print("加载 CNN 图像检索模型")
        print("=" * 60)

        if not self.model_path.exists():
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")

        # 加载 checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)

        # 获取元数据
        if 'meta' in checkpoint:
            self.meta = checkpoint['meta']
        else:
            # 默认元数据（GL18-ResNet101-GeM）
            self.meta = {
                'architecture': 'resnet101',
                'pooling': 'gem',
                'whitening': True,
                'mean': [0.485, 0.456, 0.406],
                'std': [0.229, 0.224, 0.225],
                'outputdim': 2048
            }

        # 初始化网络 (pretrained=False，因为后续会从 checkpoint 加载完整权重)
        meta_with_pretrained = self.meta.copy()
        meta_with_pretrained['pretrained'] = False
        self.model = init_network(meta_with_pretrained)

        # 加载权重
        if 'state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['state_dict'])
        else:
            self.model.load_state_dict(checkpoint)

        # 移动到设备并设置为评估模式
        if self.use_gpu:
            self.model = self.model.cuda()
        self.model.eval()

        print(f"模型加载完成: {self.model_path.name}")
        print(f"  架构: {self.meta.get('architecture', 'unknown')}")
        print(f"  池化: {self.meta.get('pooling', 'unknown')}")
        print(f"  输出维度: {self.meta.get('outputdim', 'unknown')}")
        print(f"  设备: {self.device}")

    def _load_dataset_config(self):
        """
        加载数据集配置。

        使用 core.cirtorch.datasets.testdataset.configdataset。
        """
        print("\n加载数据集配置...")

        # 检查数据目录
        if not self.data_dir.exists():
            raise FileNotFoundError(f"数据目录不存在: {self.data_dir}")

        # 加载配置
        self.cfg = configdataset(self.dataset, self.data_dir.parent)

        print(f"  数据集: {self.dataset}")
        print(f"  图像数量: {self.cfg['n']}")
        print(f"  查询数量: {self.cfg['nq']}")

    def _get_transform(self) -> transforms.Compose:
        """
        获取图像预处理变换。

        返回：
            torchvision.transforms.Compose 对象
        """
        mean = self.meta.get('mean', [0.485, 0.456, 0.406])
        std = self.meta.get('std', [0.229, 0.224, 0.225])

        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

    def extract_features(self,
                        image_paths: List[str],
                        bbxs: List[Tuple] = None) -> np.ndarray:
        """
        提取图像特征。

        参数：
            image_paths: 图像路径列表
            bbxs: 边界框列表（可选）

        返回：
            [N, D] 特征矩阵
        """
        print(f"提取特征: {len(image_paths)} 张图像...")

        transform = self._get_transform()

        # 使用 extract_vectors 提取特征（直接传递路径）
        with torch.no_grad():
            vecs = extract_vectors(
                self.model, image_paths, self.image_size, transform,
                bbxs=bbxs, ms=self.multiscale, print_freq=50
            )

        # 转置为 [N, D]
        features = vecs.numpy().T

        print(f"  特征维度: {features.shape}")
        return features

    def extract_database_features(self, cache_dir: str = None) -> np.ndarray:
        """
        提取数据库所有图像的特征（支持缓存）。

        参数：
            cache_dir: 缓存目录
                - None: 不使用缓存
                - "auto": 使用默认路径 backend/cache/cir/{dataset}
                - 其他: 使用指定路径

        返回：
            [N, D] 数据库特征矩阵
        """
        print("\n" + "=" * 60)
        print("提取数据库特征")
        print("=" * 60)

        # 处理缓存路径
        if cache_dir == "auto":
            project_root = Path(__file__).parent.parent
            cache_dir = project_root / "backend" / "cache" / "cir" / self.dataset
        elif cache_dir:
            cache_dir = Path(cache_dir)
        else:
            cache_dir = None

        # 检查缓存
        if cache_dir:
            cache_path = Path(cache_dir) / "features.npz"
            if cache_path.exists():
                print(f"加载缓存特征: {cache_path}")
                data = np.load(cache_path, allow_pickle=True)
                db_features = data.get('features')
                print(f"  特征维度: {db_features.shape}")
                return db_features

        # 获取数据库图像路径
        image_paths = [self.cfg['im_fname'](self.cfg, i) for i in range(self.cfg['n'])]

        features = self.extract_features(image_paths)

        # 保存缓存
        if cache_dir:
            cache_dir = Path(cache_dir)
            cache_path = cache_dir / "features.npz"
            cache_dir.mkdir(parents=True, exist_ok=True)
            np.savez(cache_path, features=features)
            print(f"特征已缓存: {cache_path}")

        return features

    def extract_query_features(self, cache_dir: str = None) -> Tuple[np.ndarray, List[Tuple]]:
        """
        提取查询图像的特征（支持缓存）。

        参数：
            cache_dir: 缓存目录
                - None: 不使用缓存
                - "auto": 使用默认路径 backend/cache/cir/{dataset}
                - 其他: 使用指定路径

        返回：
            (特征矩阵, 边界框列表)
        """
        print("\n" + "=" * 60)
        print("提取查询特征")
        print("=" * 60)

        # 处理缓存路径
        if cache_dir == "auto":
            project_root = Path(__file__).parent.parent
            cache_dir = project_root / "backend" / "cache" / "cir" / self.dataset
        elif cache_dir:
            cache_dir = Path(cache_dir)
        else:
            cache_dir = None

        # 检查缓存
        if cache_dir:
            cache_path = Path(cache_dir) / "query_features.npz"
            if cache_path.exists():
                print(f"加载缓存特征: {cache_path}")
                data = np.load(cache_path, allow_pickle=True)
                query_features = data.get('features')
                # 仍需获取边界框
                bbxs = []
                for i in range(self.cfg['nq']):
                    try:
                        bbx = tuple(self.cfg['gnd'][i]['bbx'])
                        bbxs.append(bbx)
                    except:
                        bbxs.append(None)
                print(f"  特征维度: {query_features.shape}")
                return query_features, bbxs

        # 获取查询图像路径
        image_paths = [self.cfg['qim_fname'](self.cfg, i) for i in range(self.cfg['nq'])]

        # 获取查询边界框
        bbxs = []
        for i in range(self.cfg['nq']):
            try:
                bbx = tuple(self.cfg['gnd'][i]['bbx'])
                bbxs.append(bbx)
            except:
                bbxs.append(None)

        features = self.extract_features(image_paths, bbxs)

        # 保存缓存
        if cache_dir:
            cache_path = Path(cache_dir) / "query_features.npz"
            cache_dir.mkdir(parents=True, exist_ok=True)
            np.savez(cache_path, features=features)
            print(f"特征已缓存: {cache_path}")

        return features, bbxs

    def evaluate_retrieval(self,
                          db_features: np.ndarray,
                          query_features: np.ndarray) -> Dict[str, Any]:
        """
        执行检索评估。

        参数：
            db_features: 数据库特征 [N, D]
            query_features: 查询特征 [Q, D]

        返回：
            评估结果字典
        """
        print("\n" + "=" * 60)
        print("执行检索评估")
        print("=" * 60)

        # 计算相似度矩阵
        print("计算相似度矩阵...")
        similarities = np.dot(query_features, db_features.T)

        # 获取排序结果（按相似度降序）
        ranks = np.argsort(-similarities, axis=1).T  # [N, Q]

        # 计算 mAP（使用标准评估协议）
        gnd = self.cfg['gnd']

        if self.dataset.startswith('oxford5k') or self.dataset.startswith('paris6k'):
            # 旧评估协议
            map_score, aps, _, _ = compute_map(ranks, gnd)
            results = {
                'mAP': float(map_score),
                'APs': aps.tolist(),
                'protocol': 'legacy'
            }
            print(f"mAP: {map_score:.4f}")

        else:
            # 新评估协议（roxford5k / rparis6k）
            # Easy 难度
            gnd_t = []
            for i in range(len(gnd)):
                g = {}
                g['ok'] = np.concatenate([gnd[i]['easy']])
                g['junk'] = np.concatenate([gnd[i]['junk'], gnd[i]['hard']])
                gnd_t.append(g)
            mapE, apsE, mprE, prsE = compute_map(ranks, gnd_t, [1, 5, 10])

            # Medium 难度
            gnd_t = []
            for i in range(len(gnd)):
                g = {}
                g['ok'] = np.concatenate([gnd[i]['easy'], gnd[i]['hard']])
                g['junk'] = np.concatenate([gnd[i]['junk']])
                gnd_t.append(g)
            mapM, apsM, mprM, prsM = compute_map(ranks, gnd_t, [1, 5, 10])

            # Hard 难度
            gnd_t = []
            for i in range(len(gnd)):
                g = {}
                g['ok'] = np.concatenate([gnd[i]['hard']])
                g['junk'] = np.concatenate([gnd[i]['junk'], gnd[i]['easy']])
                gnd_t.append(g)
            mapH, apsH, mprH, prsH = compute_map(ranks, gnd_t, [1, 5, 10])

            results = {
                'mAP_easy': float(mapE),
                'mAP_medium': float(mapM),
                'mAP_hard': float(mapH),
                'APs_easy': apsE.tolist(),
                'APs_medium': apsM.tolist(),
                'APs_hard': apsH.tolist(),
                'precision@k': {
                    'easy': mprE.tolist(),
                    'medium': mprM.tolist(),
                    'hard': mprH.tolist()
                },
                'protocol': 'new'
            }

            print(f"mAP (Easy):   {mapE:.4f}")
            print(f"mAP (Medium): {mapM:.4f}")
            print(f"mAP (Hard):   {mapH:.4f}")
            print(f"P@K (Medium): {mprM}")

        return results

    def evaluate_aspe(self,
                     db_features: np.ndarray,
                     query_features: np.ndarray,
                     map_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行 ASPE 加密验证。

        验证密文 mAP 与明文 mAP 是否一致。
        对于 roxford5k/rparis6k 数据集，计算 Easy/Medium/Hard 三种难度。

        参数：
            db_features: 数据库特征 [N, D]
            query_features: 查询特征 [Q, D]
            map_results: 明文 mAP 结果

        返回：
            ASPE 验证结果，包含三种难度的对比
        """
        print("\n" + "=" * 60)
        print("ASPE 加密验证")
        print("=" * 60)

        feature_dim = db_features.shape[1]

        # 1. 初始化 ASPE
        print(f"初始化 ASPE (feature_dim={feature_dim})...")
        aspe = ASPEForCNN(feature_dim=feature_dim, seed=42, device=str(self.device))
        aspe.generate_keys()

        # 2. 加密数据库特征
        print("加密数据库特征...")
        encrypted_db = aspe.encrypt_database(db_features)
        print(f"  加密后维度: {encrypted_db.shape}")

        # 3. 加密查询特征（生成陷阱门）
        print("加密查询特征...")
        encrypted_queries = []
        for i in range(len(query_features)):
            enc_q = aspe.encrypt_query(query_features[i])
            encrypted_queries.append(enc_q)
        encrypted_queries = torch.stack(encrypted_queries)
        print(f"  加密后维度: {encrypted_queries.shape}")

        # 4. 计算密文相似度
        print("计算密文相似度...")
        cipher_similarities = torch.matmul(encrypted_db, encrypted_queries.T).numpy()

        # 5. 获取密文排序结果
        cipher_ranks = np.argsort(-cipher_similarities, axis=0).T

        # 6. 计算密文 mAP
        gnd = self.cfg['gnd']

        if self.dataset.startswith('roxford5k') or self.dataset.startswith('rparis6k'):
            # 新评估协议：计算 Easy/Medium/Hard 三种难度

            # Easy 难度
            gnd_t = []
            for i in range(len(gnd)):
                g = {}
                g['ok'] = np.concatenate([gnd[i]['easy']])
                g['junk'] = np.concatenate([gnd[i]['junk'], gnd[i]['hard']])
                gnd_t.append(g)
            cipher_map_easy, _, _, _ = compute_map(cipher_ranks.T, gnd_t)

            # Medium 难度
            gnd_t = []
            for i in range(len(gnd)):
                g = {}
                g['ok'] = np.concatenate([gnd[i]['easy'], gnd[i]['hard']])
                g['junk'] = np.concatenate([gnd[i]['junk']])
                gnd_t.append(g)
            cipher_map_medium, _, _, _ = compute_map(cipher_ranks.T, gnd_t)

            # Hard 难度
            gnd_t = []
            for i in range(len(gnd)):
                g = {}
                g['ok'] = np.concatenate([gnd[i]['hard']])
                g['junk'] = np.concatenate([gnd[i]['junk'], gnd[i]['easy']])
                gnd_t.append(g)
            cipher_map_hard, _, _, _ = compute_map(cipher_ranks.T, gnd_t)

            # 获取明文 mAP
            plaintext_map_easy = map_results.get('mAP_easy', 0)
            plaintext_map_medium = map_results.get('mAP_medium', 0)
            plaintext_map_hard = map_results.get('mAP_hard', 0)

            # 计算误差
            error_easy = abs(cipher_map_easy - plaintext_map_easy)
            error_medium = abs(cipher_map_medium - plaintext_map_medium)
            error_hard = abs(cipher_map_hard - plaintext_map_hard)

            print("\n结果对比:")
            print(f"  Easy   - 明文: {plaintext_map_easy:.6f}, 密文: {cipher_map_easy:.6f}, 误差: {error_easy:.2e}")
            print(f"  Medium - 明文: {plaintext_map_medium:.6f}, 密文: {cipher_map_medium:.6f}, 误差: {error_medium:.2e}")
            print(f"  Hard   - 明文: {plaintext_map_hard:.6f}, 密文: {cipher_map_hard:.6f}, 误差: {error_hard:.2e}")

            results = {
                'plaintext': {
                    'mAP_easy': float(plaintext_map_easy),
                    'mAP_medium': float(plaintext_map_medium),
                    'mAP_hard': float(plaintext_map_hard)
                },
                'ciphertext': {
                    'mAP_easy': float(cipher_map_easy),
                    'mAP_medium': float(cipher_map_medium),
                    'mAP_hard': float(cipher_map_hard)
                },
                'error': {
                    'easy': float(error_easy),
                    'medium': float(error_medium),
                    'hard': float(error_hard)
                },
                'protocol': 'new'
            }

        else:
            # 旧评估协议（oxford5k / paris6k）
            cipher_map, _, _, _ = compute_map(cipher_ranks.T, gnd)
            plaintext_map = map_results.get('mAP', 0)
            error = abs(cipher_map - plaintext_map)

            print("\n结果对比:")
            print(f"  明文 mAP: {plaintext_map:.6f}")
            print(f"  密文 mAP: {cipher_map:.6f}")
            print(f"  误差:     {error:.2e}")

            results = {
                'plaintext': {
                    'mAP': float(plaintext_map)
                },
                'ciphertext': {
                    'mAP': float(cipher_map)
                },
                'error': float(error),
                'protocol': 'legacy'
            }

        # 7. 验证内积保持性（使用真实特征数据）
        print("验证内积保持性...")
        # 将 numpy 数组转换为 torch.Tensor 以便传入验证方法
        db_tensor = torch.from_numpy(db_features) if isinstance(db_features, np.ndarray) else db_features
        query_tensor = torch.from_numpy(query_features) if isinstance(query_features, np.ndarray) else query_features
        verification = aspe.verify_inner_product_preservation(
            db_features=db_tensor,
            query_features=query_tensor,
            num_samples=min(10, len(query_features))
        )
        results['inner_product_preserved'] = verification['passed']
        results['verification_details'] = verification

        print(f"  数据来源: {verification.get('data_source', 'unknown')}")
        print(f"  样本数量: {verification.get('num_samples', 0)}")
        print(f"  平均误差: {verification.get('mean_absolute_diff', 0):.2e}")
        print(f"  最大误差: {verification.get('max_absolute_diff', 0):.2e}")
        print(f"  内积保持: {'通过' if verification['passed'] else '失败'}")

        # 8. 保存加密特征到后端缓存
        print("保存加密特征到缓存...")
        cache_service = HashCacheService()
        if isinstance(encrypted_db, torch.Tensor):
            encrypted_np = encrypted_db.cpu().numpy()
        else:
            encrypted_np = encrypted_db
        cache_service.save_cir_encrypted(encrypted_np, self.dataset)

        return results

    def generate_visualizations(self,
                               map_results: Dict,
                               aspe_results: Optional[Dict] = None,
                               output_dir: str = '.') -> List[str]:
        """
        生成可视化图表。

        参数：
            map_results: mAP 结果
            aspe_results: ASPE 验证结果（可选）
            output_dir: 输出目录

        返回：
            图表文件路径列表
        """
        print("\n" + "=" * 60)
        print("生成可视化图表")
        print("=" * 60)

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        chart_paths = []

        # 1. mAP 对比图
        print("生成 mAP 对比图...")
        if map_results.get('protocol') == 'new':
            # 使用 CIR 专用可视化函数（Easy/Medium/Hard）
            map_path = str(output_dir / 'cir_map.png')
            plot_cir_map_comparison(map_results, map_path, title='CNN 图像检索性能')
        else:
            # 旧评估协议
            map_data = {'mAP': map_results.get('mAP', 0)}
            map_path = str(output_dir / 'cir_map.png')
            plot_map_comparison(map_data, map_path, title='CNN 图像检索性能')

        chart_paths.append(map_path)
        print(f"  已保存: {map_path}")

        # 2. ASPE 对比图（如果有 ASPE 结果）
        if aspe_results:
            print("生成 ASPE 对比图...")
            aspe_path = str(output_dir / 'cir_aspe_comparison.png')
            # 使用 CIR 专用的 ASPE 可视化函数
            plot_cir_aspe_comparison(aspe_results, aspe_path)
            chart_paths.append(aspe_path)
            print(f"  已保存: {aspe_path}")

        return chart_paths

    def generate_report(self,
                       map_results: Dict,
                       aspe_results: Optional[Dict] = None,
                       chart_paths: List[str] = None,
                       output_dir: str = '.') -> str:
        """
        生成 Markdown 评估报告。

        参数：
            map_results: mAP 结果
            aspe_results: ASPE 验证结果（可选）
            chart_paths: 图表路径列表
            output_dir: 输出目录

        返回：
            报告文件路径
        """
        print("\n" + "=" * 60)
        print("生成评估报告")
        print("=" * 60)

        output_dir = Path(output_dir)
        report_path = output_dir / 'cir_evaluation_report.md'

        report_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        lines = [
            "# CNN 图像检索评估报告",
            "",
            f"**生成时间**: {report_date}",
            f"**数据集**: {self.dataset}",
            f"**模型**: {self.model_path.name}",
            f"**特征维度**: {self.meta.get('outputdim', 'N/A')}",
            "",
            "---",
            "",
            "## 1. 检索性能评估",
            "",
        ]

        if map_results.get('protocol') == 'new':
            lines.extend([
                "### 1.1 新评估协议结果",
                "",
                "| 难度 | mAP |",
                "|------|-----|",
                f"| Easy   | {map_results.get('mAP_easy', 0):.4f} |",
                f"| Medium | {map_results.get('mAP_medium', 0):.4f} |",
                f"| Hard   | {map_results.get('mAP_hard', 0):.4f} |",
                "",
                "### 1.2 Precision@K",
                "",
                "| K | Easy | Medium | Hard |",
                "|---|------|--------|------|",
            ])
            pr = map_results.get('precision@k', {})
            for i, k in enumerate([1, 5, 10]):
                easy = pr.get('easy', [0, 0, 0])[i] if pr else 0
                medium = pr.get('medium', [0, 0, 0])[i] if pr else 0
                hard = pr.get('hard', [0, 0, 0])[i] if pr else 0
                lines.append(f"| {k} | {easy:.4f} | {medium:.4f} | {hard:.4f} |")
        else:
            lines.extend([
                "### 1.1 旧评估协议结果",
                "",
                f"- **mAP**: {map_results.get('mAP', 0):.4f}",
                "",
            ])

        # ASPE 验证结果
        if aspe_results:
            lines.extend([
                "",
                "---",
                "",
                "## 2. ASPE 加密验证",
                "",
                "### 2.1 验证背景",
                "",
                "ASPE (Asymmetric Scalar-product-preserving Encryption) 是一种非对称标量积保持加密方案，",
                "用于保护图像检索中的数据隐私。其核心特性包括：",
                "",
                "- **内积保持性**: 密文内积保持明文内积的比例关系",
                "- **排序不变性**: 加密前后的检索排序结果一致",
                "- **mAP 等价性**: 密文检索的 mAP 应与明文相等",
                "",
            ])

            # 根据协议类型生成不同的报告内容
            if aspe_results.get('protocol') == 'new':
                # 新评估协议：三种难度对比
                lines.extend([
                    "### 2.2 明文 vs 密文 mAP 对比",
                    "",
                    "| 难度 | 明文 mAP | 密文 mAP | 误差 |",
                    "|------|----------|----------|------|",
                    f"| Easy   | {aspe_results.get('plaintext', {}).get('mAP_easy', 0):.6f} | {aspe_results.get('ciphertext', {}).get('mAP_easy', 0):.6f} | {aspe_results.get('error', {}).get('easy', 0):.2e} |",
                    f"| Medium | {aspe_results.get('plaintext', {}).get('mAP_medium', 0):.6f} | {aspe_results.get('ciphertext', {}).get('mAP_medium', 0):.6f} | {aspe_results.get('error', {}).get('medium', 0):.2e} |",
                    f"| Hard   | {aspe_results.get('plaintext', {}).get('mAP_hard', 0):.6f} | {aspe_results.get('ciphertext', {}).get('mAP_hard', 0):.6f} | {aspe_results.get('error', {}).get('hard', 0):.2e} |",
                    "",
                    "### 2.3 内积保持验证",
                    "",
                    f"- **验证结果**: {'✅ 通过' if aspe_results.get('inner_product_preserved') else '❌ 失败'}",
                    "",
                    "### 2.4 结论",
                    "",
                ])

                # 判断是否通过验证
                errors = aspe_results.get('error', {})
                max_error = max(errors.get('easy', 1), errors.get('medium', 1), errors.get('hard', 1))
                if aspe_results.get('inner_product_preserved') and max_error < 1e-3:
                    lines.append("✅ ASPE 加密验证通过，密文检索与明文检索结果一致。")
                else:
                    lines.append("❌ ASPE 加密验证存在问题，需要检查加密实现。")

            else:
                # 旧评估协议：单一 mAP
                lines.extend([
                    "### 2.2 验证结果",
                    "",
                    "| 指标 | 值 |",
                    "|------|-----|",
                    f"| 明文 mAP | {aspe_results.get('plaintext', {}).get('mAP', 0):.6f} |",
                    f"| 密文 mAP | {aspe_results.get('ciphertext', {}).get('mAP', 0):.6f} |",
                    f"| 误差 | {aspe_results.get('error', 0):.2e} |",
                    f"| 内积保持 | {'通过' if aspe_results.get('inner_product_preserved') else '失败'} |",
                    "",
                    "### 2.3 结论",
                    "",
                ])
                if aspe_results.get('inner_product_preserved') and aspe_results.get('error', 1) < 1e-3:
                    lines.append("ASPE 加密验证通过，密文检索与明文检索结果一致。")
                else:
                    lines.append("ASPE 加密验证存在问题，需要检查加密实现。")

        lines.extend([
            "",
            "---",
            "",
            "## 3. 配置信息",
            "",
            f"- 数据集: {self.dataset}",
            f"- 图像数量: {self.cfg['n']}",
            f"- 查询数量: {self.cfg['nq']}",
            f"- 模型架构: {self.meta.get('architecture', 'N/A')}",
            f"- 池化方法: {self.meta.get('pooling', 'N/A')}",
            f"- 图像大小: {self.image_size}",
            f"- 多尺度: {self.multiscale}",
            "",
            "---",
            "",
            "*本报告由 CIREvaluator 自动生成*",
        ])

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

        print(f"  已保存: {report_path}")
        return str(report_path)

    def evaluate(self,
                run_aspe: bool = True,
                output_dir: str = None,
                use_cache: bool = True) -> Dict[str, Any]:
        """
        执行完整评估。

        参数：
            run_aspe: 是否执行 ASPE 加密验证（默认 True）
            output_dir: 输出目录（默认为当前目录）
            use_cache: 是否使用特征缓存（缓存路径：backend/cache/cir/{dataset}）

        返回：
            所有评估结果
        """
        print("\n" + "=" * 60)
        print("CNN 图像检索完整评估")
        print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)

        if output_dir is None:
            output_dir = f"results/cir_{self.dataset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # 1. 提取特征（支持缓存，缓存路径为 backend/cache/cir/{dataset}）
        db_features = self.extract_database_features(cache_dir="auto" if use_cache else None)
        query_features, bbxs = self.extract_query_features(cache_dir="auto" if use_cache else None)

        # 2. 检索评估
        map_results = self.evaluate_retrieval(db_features, query_features)

        # 3. ASPE 加密验证
        aspe_results = None
        if run_aspe:
            aspe_results = self.evaluate_aspe(db_features, query_features, map_results)

        # 4. 生成可视化
        chart_paths = self.generate_visualizations(map_results, aspe_results, output_dir)

        # 5. 生成报告
        report_path = self.generate_report(map_results, aspe_results, chart_paths, output_dir)

        # 6. 汇总结果
        all_results = {
            'config': {
                'model_path': str(self.model_path),
                'data_dir': str(self.data_dir),
                'dataset': self.dataset,
                'image_size': self.image_size,
                'multiscale': self.multiscale,
                'evaluated_at': datetime.now().isoformat()
            },
            'map_results': map_results,
            'aspe_results': aspe_results,
            'chart_paths': chart_paths,
            'report_path': report_path
        }

        # 保存 JSON 结果（先清理 NaN，再使用 NaN 感知编码器）
        json_path = Path(output_dir) / 'cir_results.json'
        json_path.parent.mkdir(parents=True, exist_ok=True)
        cleaned_results = _clean_nans(all_results)
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(cleaned_results, f, indent=2, ensure_ascii=False, cls=NaNAwareJSONEncoder)
        print(f"\nJSON 结果已保存: {json_path}")

        # 总结
        print("\n" + "=" * 60)
        print("评估完成")
        print("=" * 60)

        if map_results.get('protocol') == 'new':
            print(f"mAP (Medium): {map_results.get('mAP_medium', 0):.4f}")
        else:
            print(f"mAP: {map_results.get('mAP', 0):.4f}")

        print(f"图表数量: {len(chart_paths)}")
        print(f"报告路径: {report_path}")

        return all_results


def main():
    """命令行入口。"""
    import argparse

    parser = argparse.ArgumentParser(description='CNN 图像检索评估器')
    parser.add_argument('--model_path', type=str, required=True,
                       help='预训练模型路径')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='数据集目录')
    parser.add_argument('--dataset', type=str, default='roxford5k',
                       choices=['oxford5k', 'paris6k', 'roxford5k', 'rparis6k'],
                       help='数据集名称')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='输出目录')
    parser.add_argument('--image_size', type=int, default=1024,
                       help='图像缩放大小')
    parser.add_argument('--no-gpu', action='store_true',
                       help='禁用 GPU')
    parser.add_argument('--no-aspe', action='store_true',
                       help='禁用 ASPE 加密验证（默认启用）')
    parser.add_argument('--no-cache', action='store_true',
                       help='禁用特征缓存')

    args = parser.parse_args()

    evaluator = CIREvaluator(
        model_path=args.model_path,
        data_dir=args.data_dir,
        dataset=args.dataset,
        use_gpu=not args.no_gpu,
        image_size=args.image_size
    )

    results = evaluator.evaluate(
        run_aspe=not args.no_aspe,
        output_dir=args.output_dir,
        use_cache=not args.no_cache
    )

    print("\n评估完成！")


if __name__ == "__main__":
    main()