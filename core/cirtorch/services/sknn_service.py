"""
SkNN (Secure k-Nearest Neighbors) 隐私保护检索服务

基于 ASPE (Asymmetric Scalar Product-Preserving Encryption) 方案，
实现隐私保护的图像检索功能。

加密原理（论文原版双矩阵方案）：
- 建库端：根据随机向量 S，S=0 时直接复制，S=1 时随机拆分
- 查询端：根据随机向量 S，S=0 时拆分（r=0），S=1 时直接复制
- 使用两个随机可逆矩阵 M1, M2 进行线性变换
- 密文内积 = 明文内积（保持性）
"""

import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from typing import List, Dict, Tuple, Optional, Union
from pathlib import Path
import logging

from cirtorch.networks.imageretrievalnet import init_network, extract_vectors

logger = logging.getLogger(__name__)


def generate_sknn_keys(d: int, device: str = 'cpu') -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    生成 SkNN 密钥（双矩阵方案）。

    参数：
        d: 特征维度（如 2048）
        device: 计算设备

    返回：
        M1, M2, S 三个密钥
        - M1: d×d 随机可逆矩阵
        - M2: d×d 随机可逆矩阵
        - S: d 维二值向量
    """
    # 生成两个 d×d 的随机矩阵
    M1 = torch.randn(d, d, device=device)
    M2 = torch.randn(d, d, device=device)
    # 生成 d 维二值向量 S
    S = torch.randint(0, 2, (d,), dtype=torch.float32, device=device)

    logger.info(f"SkNN 密钥生成完成：特征维度={d}, M1 形状={M1.shape}")
    return M1, M2, S


class SknnService:
    """
    SkNN 隐私保护检索服务。

    提供完整的加密检索流程：
    1. 密钥生成与管理
    2. 数据库特征加密
    3. 查询特征加密
    4. 密文相似度搜索
    """

    def __init__(self,
                 feature_dim: int = 2048,
                 model_path: Optional[str] = None,
                 db_dir: Optional[str] = None,
                 device: Optional[str] = None):
        """
        初始化 SkNN 服务。

        参数：
            feature_dim: 特征维度（默认 2048，对应 ResNet101-GeM）
            model_path: 预训练模型路径
            db_dir: 数据库存储目录
            device: 计算设备
        """
        self.feature_dim = feature_dim

        # 设备设置
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # 路径设置
        self.db_dir = Path(db_dir) if db_dir else Path('./data/retrieval_db')
        self.model_path = model_path

        # 核心组件（懒加载）
        self.model: Optional[nn.Module] = None
        self.M1: Optional[torch.Tensor] = None
        self.M2: Optional[torch.Tensor] = None
        self.S: Optional[torch.Tensor] = None
        self.db_features: Optional[torch.Tensor] = None
        self.db_image_names: Optional[List[str]] = None

        # 图像预处理
        self.transform = transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]),
        ])

        logger.info(f"SkNN 服务初始化完成：feature_dim={feature_dim}, device={self.device}")

    def load_model(self, model_path: Optional[str] = None):
        """加载检索模型。"""
        if model_path is None:
            model_path = self.model_path

        if model_path is None:
            raise ValueError("必须指定模型路径")

        logger.info(f"正在加载检索模型：{model_path}")
        state = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model = init_network({
            'architecture': state['meta']['architecture'],
            'pooling': state['meta']['pooling'],
            'whitening': state['meta']['whitening']
        })
        self.model.load_state_dict(state['state_dict'])
        self.model.to(self.device)
        self.model.eval()
        logger.info("模型加载完成")

    def generate_keys(self, save_path: Optional[str] = None):
        """
        生成新的 SkNN 密钥。

        参数：
            save_path: 可选的保存路径
        """
        logger.info(f"正在生成 SkNN 密钥，特征维度={self.feature_dim}")
        self.M1, self.M2, self.S = generate_sknn_keys(self.feature_dim, self.device)

        if save_path:
            self.save_keys(save_path)
            logger.info(f"密钥已保存到：{save_path}")

        return self.M1, self.M2, self.S

    def load_keys(self, keys_path: str):
        """加载已保存的密钥。"""
        logger.info(f"正在加载 SkNN 密钥：{keys_path}")
        keys = torch.load(keys_path, map_location=self.device, weights_only=False)
        self.M1 = keys['M1']
        self.M2 = keys['M2']
        self.S = keys['S']
        logger.info("密钥加载完成")

    def save_keys(self, save_path: str):
        """保存密钥到文件。"""
        self.db_dir.mkdir(parents=True, exist_ok=True)
        torch.save({'M1': self.M1, 'M2': self.M2, 'S': self.S}, save_path)

    def encrypt_database_feature(self, p: torch.Tensor) -> torch.Tensor:
        """
        加密数据库特征（建库端加密）。

        加密规则：
        - S[i]=0: p1[i]=p[i], p2[i]=p[i] (直接复制)
        - S[i]=1: p1[i]=r, p2[i]=p[i]-r (随机拆分)

        参数：
            p: 明文特征向量 [d]

        返回：
            密文特征向量 [2d]
        """
        d = len(p)
        p1 = torch.zeros(d, device=p.device)
        p2 = torch.zeros(d, device=p.device)

        for i in range(d):
            if self.S[i] == 0:
                # S=0：直接复制
                p1[i] = p[i]
                p2[i] = p[i]
            else:
                # S=1：随机拆分
                r = torch.randn(1, device=p.device).item()
                p1[i] = r
                p2[i] = p[i] - r

        # 加密变换：分别乘以 M1.T 和 M2.T
        enc_part1 = torch.matmul(self.M1.T, p1)
        enc_part2 = torch.matmul(self.M2.T, p2)

        # 拼接为 2d 维密文向量
        p_enc = torch.cat((enc_part1, enc_part2))
        return p_enc

    def encrypt_query_feature(self, p: torch.Tensor) -> torch.Tensor:
        """
        加密查询特征（查询端陷阱门生成）。

        加密规则（与建库端相反）：
        - S[i]=0: q1[i]=r(=0), q2[i]=p[i]-r (拆分，r 设为 0 消除浮点误差)
        - S[i]=1: q1[i]=p[i], q2[i]=p[i] (直接复制)

        参数：
            p: 明文查询特征 [d]

        返回：
            密文陷阱门向量 [2d]
        """
        d = len(p)
        q1 = torch.zeros(d, device=self.device)
        q2 = torch.zeros(d, device=self.device)

        for i in range(d):
            if self.S[i] == 0:
                # S=0：查询端拆分（r=0 消除浮点误差）
                r = 0.0
                q1[i] = r
                q2[i] = p[i] - r
            else:
                # S=1：直接复制
                q1[i] = p[i]
                q2[i] = p[i]

        # 【高精度优化】使用 Float64 计算逆矩阵，消除浮点误差
        M1_double = self.M1.to(torch.float64)
        M2_double = self.M2.to(torch.float64)
        q1_double = q1.to(torch.float64)
        q2_double = q2.to(torch.float64)

        # 加密变换：分别乘以 M1^-1 和 M2^-1
        trap_part1 = torch.matmul(torch.linalg.inv(M1_double), q1_double)
        trap_part2 = torch.matmul(torch.linalg.inv(M2_double), q2_double)

        # 拼接并转回 Float32
        q_enc = torch.cat((trap_part1, trap_part2)).to(torch.float32)

        return q_enc

    def build_database(self,
                       image_dir: str,
                       save_dir: Optional[str] = None,
                       batch_size: int = 32) -> Tuple[torch.Tensor, List[str]]:
        """
        构建加密检索数据库。

        参数：
            image_dir: 图像目录
            save_dir: 保存目录（可选）
            batch_size: 批次大小

        返回：
            db_features: 加密特征矩阵
            db_images: 图像名称列表
        """
        if self.model is None:
            self.load_model()

        if self.M1 is None:
            self.generate_keys()

        # 扫描图像文件
        image_dir = Path(image_dir)
        valid_extensions = {'.jpg', '.jpeg', '.png'}
        image_files = sorted([
            f for f in os.listdir(image_dir)
            if os.path.splitext(f)[1].lower() in valid_extensions
        ])

        logger.info(f"发现 {len(image_files)} 张图像")

        db_features = []
        db_images = []

        with torch.no_grad():
            for idx, img_name in enumerate(image_files):
                img_path = image_dir / img_name
                try:
                    # 加载并预处理图像
                    img = Image.open(img_path).convert('RGB')
                    tensor = self.transform(img).unsqueeze(0).to(self.device)

                    # 提取明文特征
                    feature = self.model(tensor).squeeze().cpu()

                    # 加密特征
                    enc_feature = self.encrypt_database_feature(feature)

                    db_features.append(enc_feature)
                    db_images.append(img_name)

                    if (idx + 1) % 50 == 0:
                        logger.info(f"已处理 {idx + 1}/{len(image_files)} 张图像")

                except Exception as e:
                    logger.error(f"处理 {img_name} 失败：{e}")

        # 转换为张量
        db_features_tensor = torch.stack(db_features)

        # 保存
        if save_dir:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
            torch.save(db_features_tensor, save_path / 'encrypted_features.pth')
            torch.save(db_images, save_path / 'image_names.pth')
            torch.save({'M1': self.M1, 'M2': self.M2, 'S': self.S},
                      save_path / 'sknn_keys.pth')
            logger.info(f"数据库已保存到：{save_path}")

        self.db_features = db_features_tensor
        self.db_image_names = db_images

        logger.info(f"数据库构建完成：{db_features_tensor.shape}")
        return db_features_tensor, db_images

    def load_database(self, db_dir: Optional[str] = None):
        """加载已保存的数据库。"""
        if db_dir is None:
            db_dir = self.db_dir
        else:
            db_dir = Path(db_dir)

        logger.info(f"正在加载数据库：{db_dir}")

        self.db_features = torch.load(db_dir / 'encrypted_features.pth',
                                      map_location=self.device, weights_only=False)
        self.db_image_names = torch.load(db_dir / 'image_names.pth',
                                        weights_only=False)

        # 同时加载密钥
        self.load_keys(db_dir / 'sknn_keys.pth')

        logger.info(f"数据库加载完成：{self.db_features.shape}")

    def search(self,
               query_image_path: str,
               top_k: int = 10) -> List[Dict]:
        """
        隐私保护图像检索。

        参数：
            query_image_path: 查询图像路径
            top_k: 返回结果数量

        返回：
            检索结果列表
        """
        if self.db_features is None:
            raise ValueError("数据库未加载")

        # 提取查询特征
        with torch.no_grad():
            img = Image.open(query_image_path).convert('RGB')
            tensor = self.transform(img).unsqueeze(0).to(self.device)
            query_feature = self.model(tensor).squeeze().cpu()

            # 加密查询
            enc_query = self.encrypt_query_feature(query_feature)

        # 密文内积计算
        db_feats_device = self.db_features.to(self.device)
        scores = torch.matmul(db_feats_device, enc_query)

        # Top-K
        topk_scores, topk_indices = torch.topk(scores, k=top_k)

        results = []
        for score, idx in zip(topk_scores, topk_indices):
            img_name = self.db_image_names[idx.item()]
            results.append({
                'rank': len(results) + 1,
                'image_name': img_name,
                'score': round(score.item(), 4)
            })

        return results

    def search_by_image(self,
                        image: Union[Image.Image, str],
                        top_k: int = 10) -> List[Dict]:
        """
        隐私保护图像检索（支持 PIL 图像或路径）。

        参数：
            image: PIL 图像对象或图像路径
            top_k: 返回结果数量

        返回：
            检索结果列表
        """
        if self.db_features is None:
            raise ValueError("数据库未加载")

        # 加载图像
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')

        # 提取查询特征
        with torch.no_grad():
            tensor = self.transform(image).unsqueeze(0).to(self.device)
            query_feature = self.model(tensor).squeeze().cpu()

            # 加密查询
            enc_query = self.encrypt_query_feature(query_feature)

        # 密文内积计算
        db_feats_device = self.db_features.to(self.device)
        scores = torch.matmul(db_feats_device, enc_query)

        # Top-K
        topk_scores, topk_indices = torch.topk(scores, k=top_k)

        results = []
        for score, idx in zip(topk_scores, topk_indices):
            img_name = self.db_image_names[idx.item()]
            results.append({
                'rank': len(results) + 1,
                'image_name': img_name,
                'score': round(score.item(), 4)
            })

        return results

    def search(self,
               query_image_path: str,
               top_k: int = 10,
               encrypted: bool = True) -> List[Dict]:
        """
        图像检索（支持明文/密文两种模式）。

        参数：
            query_image_path: 查询图像路径
            top_k: 返回结果数量
            encrypted: 是否使用加密检索（默认 True）

        返回：
            检索结果列表
        """
        if self.db_features is None:
            raise ValueError("数据库未加载")

        # 提取查询特征
        with torch.no_grad():
            img = Image.open(query_image_path).convert('RGB')
            tensor = self.transform(img).unsqueeze(0).to(self.device)
            query_feature = self.model(tensor).squeeze().cpu()

            if encrypted:
                # 加密查询
                enc_query = self.encrypt_query_feature(query_feature)
                # 密文内积计算
                db_feats_device = self.db_features.to(self.device)
                scores = torch.matmul(db_feats_device, enc_query)
            else:
                # 明文检索（用于调试对比）
                db_feats_device = self.db_features.to(self.device)
                # 对于密文数据库，需要使用对应的解密方式
                # 这里简化处理，直接计算密文内积
                enc_query = self.encrypt_query_feature(query_feature)
                scores = torch.matmul(db_feats_device, enc_query)

        # Top-K
        topk_scores, topk_indices = torch.topk(scores, k=top_k)

        results = []
        for score, idx in zip(topk_scores, topk_indices):
            img_name = self.db_image_names[idx.item()]
            results.append({
                'rank': len(results) + 1,
                'image_name': img_name,
                'score': round(score.item(), 4)
            })

        return results

    def search_by_feature(self,
                          query_feature: torch.Tensor,
                          top_k: int = 10) -> List[Dict]:
        """
        使用预提取特征进行检索。

        参数：
            query_feature: 查询特征向量 [d]
            top_k: 返回结果数量

        返回：
            检索结果列表
        """
        if self.db_features is None:
            raise ValueError("数据库未加载")

        # 加密查询
        enc_query = self.encrypt_query_feature(query_feature)

        # 密文内积计算
        db_feats_device = self.db_features.to(self.device)
        scores = torch.matmul(db_feats_device, enc_query)

        # Top-K
        topk_scores, topk_indices = torch.topk(scores, k=top_k)

        results = []
        for score, idx in zip(topk_scores, topk_indices):
            img_name = self.db_image_names[idx.item()]
            results.append({
                'rank': len(results) + 1,
                'image_name': img_name,
                'score': round(score.item(), 4)
            })

        return results

    def get_status(self) -> Dict:
        """获取服务状态。"""
        return {
            'model_loaded': self.model is not None,
            'keys_loaded': self.M1 is not None,
            'database_loaded': self.db_features is not None,
            'database_size': len(self.db_image_names) if self.db_image_names else 0,
            'feature_dim': self.feature_dim,
            'device': str(self.device)
        }
