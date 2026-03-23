"""CNN Image Retrieval Service for Backend API.

基于 core/cirtorch 模块，提供 CNN 图像检索服务。
支持两种检索模式：
1. 明文检索（传统 CNN 特征相似度）
2. 隐私检索（ASPE 加密保护）
"""

import ssl
# 解决 SSL 证书验证问题，允许下载预训练模型
ssl._create_default_https_context = ssl._create_unverified_context

from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import os
from pathlib import Path
import sys
import torch
import numpy as np
from PIL import Image

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.cirtorch.networks.imageretrievalnet import init_network, extract_vectors
from core.aspe.cnn_wrapper import ASPEForCNN
from torchvision import transforms


@dataclass
class ImageSearchResult:
    """Image search result for API response."""
    image_id: str
    image_name: str
    score: float
    rank: int


class CIRService:
    """
    Backend service for CNN Image Retrieval.

    This service wraps the core CIR functionality for use in the FastAPI backend.
    支持明文和加密两种检索模式。

    架构设计：
    - 使用 core.cirtorch.networks 加载 CNN 模型和提取特征
    - 使用 core.aspe.cnn_wrapper 进行 ASPE 加密
    """

    _instance: Optional['CIRService'] = None
    _initialized: bool = False

    def __init__(self,
                 feature_dim: int = 2048,
                 model_path: Optional[str] = None,
                 db_dir: Optional[str] = None,
                 device: Optional[str] = None):
        """
        Initialize the CIR service.

        Args:
            feature_dim: 特征维度（默认 2048，对应 ResNet101-GeM）
            model_path: 预训练模型路径
            db_dir: 数据库目录
            device: 计算设备
        """
        self.feature_dim = feature_dim
        self.device = torch.device(device) if device else torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )

        # CNN 模型
        self.model = None
        self.model_meta = None

        # ASPE 加密
        self.aspe = ASPEForCNN(feature_dim=feature_dim, device=str(self.device))

        # 数据库
        self.db_features: Optional[torch.Tensor] = None
        self.db_image_names: Optional[List[str]] = []
        self.db_dir = Path(db_dir) if db_dir else None

    @classmethod
    def get_instance(cls) -> 'CIRService':
        """Get singleton instance of the service."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def load_model(self, model_path: str):
        """
        加载预训练 CNN 检索模型。

        Args:
            model_path: 模型文件路径
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")

        print(f"[CIRService] 加载模型: {model_path}")

        # 加载模型权重
        state = torch.load(model_path, map_location=self.device, weights_only=False)

        # 解析模型参数
        net_params = {
            'architecture': state['meta']['architecture'],
            'pooling': state['meta']['pooling'],
            'local_whitening': state['meta'].get('local_whitening', False),
            'regional': state['meta'].get('regional', False),
            'whitening': state['meta'].get('whitening', False),
            'mean': state['meta']['mean'],
            'std': state['meta']['std'],
            'pretrained': False
        }

        # 初始化网络
        self.model = init_network(net_params)
        self.model.load_state_dict(state['state_dict'])
        self.model.to(self.device)
        self.model.eval()

        # 保存元信息
        self.model_meta = state['meta']
        if 'Lw' in state['meta']:
            self.model.meta['Lw'] = state['meta']['Lw']

        # 更新特征维度
        self.feature_dim = self.model.meta['outputdim']
        self.aspe = ASPEForCNN(feature_dim=self.feature_dim, device=str(self.device))

        print(f"[CIRService] 模型加载成功: {self.model_meta['architecture']}-{self.model_meta['pooling']}")

    def extract_features(self,
                         image_paths: List[str],
                         image_size: int = 1024) -> torch.Tensor:
        """
        批量提取图像特征。

        Args:
            image_paths: 图像路径列表
            image_size: 图像尺寸

        Returns:
            特征矩阵 [d, N]
        """
        if self.model is None:
            raise RuntimeError("模型未加载，请先调用 load_model()")

        # 构建预处理变换
        normalize = transforms.Normalize(
            mean=self.model.meta['mean'],
            std=self.model.meta['std']
        )
        transform = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])

        # 提取特征
        features = extract_vectors(
            self.model, image_paths, image_size, transform
        )

        return features

    def build_database(self,
                       image_dir: str,
                       save_dir: Optional[str] = None) -> Tuple[torch.Tensor, List[str]]:
        """
        构建加密图像数据库。

        Args:
            image_dir: 图像目录
            save_dir: 保存目录

        Returns:
            (加密特征矩阵, 图像名称列表)
        """
        if self.model is None:
            raise RuntimeError("模型未加载，请先调用 load_model()")

        print(f"[CIRService] 构建数据库: {image_dir}")

        # 收集图像文件
        image_dir = Path(image_dir)
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_paths = [
            str(p) for p in image_dir.rglob('*')
            if p.suffix.lower() in image_extensions
        ]

        if not image_paths:
            raise ValueError(f"图像目录为空: {image_dir}")

        print(f"[CIRService] 发现 {len(image_paths)} 张图像")

        # 提取特征
        features = self.extract_features(image_paths)
        features_np = features.numpy()

        # 生成密钥
        self.aspe.generate_keys()

        # 加密特征
        print("[CIRService] 加密特征...")
        encrypted_features = self.aspe.encrypt_database(features_np.T)  # [N, 2d]

        # 保存数据库
        self.db_features = encrypted_features
        self.db_image_names = [Path(p).name for p in image_paths]

        if save_dir:
            self.save_database(save_dir)

        print(f"[CIRService] 数据库构建完成: {len(self.db_image_names)} 张图像")

        return self.db_features, self.db_image_names

    def search(self,
               query_image_path: str,
               top_k: int = 10) -> List[ImageSearchResult]:
        """
        在加密数据库中搜索相似图像。

        Args:
            query_image_path: 查询图像路径
            top_k: 返回结果数量

        Returns:
            搜索结果列表
        """
        if self.model is None:
            raise RuntimeError("模型未加载")

        if self.db_features is None:
            raise RuntimeError("数据库未构建")

        if not os.path.exists(query_image_path):
            raise FileNotFoundError(f"查询图像不存在: {query_image_path}")

        # 提取查询特征
        query_features = self.extract_features([query_image_path])
        query_feature = query_features[:, 0]  # [d]

        # 加密查询特征
        encrypted_query = self.aspe.encrypt_query(query_feature)

        # 计算密文内积
        scores = torch.matmul(self.db_features, encrypted_query)

        # Top-K 排序
        top_k = min(top_k, len(self.db_image_names))
        topk_scores, topk_indices = torch.topk(scores, k=top_k)

        # 构建结果
        results = []
        for rank, (score, idx) in enumerate(zip(topk_scores.tolist(), topk_indices.tolist())):
            results.append(ImageSearchResult(
                image_id=str(idx),
                image_name=self.db_image_names[idx],
                score=float(score),
                rank=rank + 1
            ))

        return results

    def save_database(self, save_dir: str):
        """保存加密数据库到磁盘。"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        torch.save(self.db_features, save_dir / 'encrypted_features.pth')
        torch.save(self.db_image_names, save_dir / 'image_names.pth')
        self.aspe.save_keys(save_dir / 'aspe_keys.pth')

        self.db_dir = save_dir
        print(f"[CIRService] 数据库已保存: {save_dir}")

    def load_database(self, db_dir: str):
        """从磁盘加载加密数据库。"""
        db_dir = Path(db_dir)

        if not (db_dir / 'encrypted_features.pth').exists():
            raise FileNotFoundError(f"数据库文件不存在: {db_dir}")

        self.db_features = torch.load(db_dir / 'encrypted_features.pth',
                                       map_location=self.device, weights_only=False)
        self.db_image_names = torch.load(db_dir / 'image_names.pth',
                                          weights_only=False)
        self.aspe.load_keys(db_dir / 'aspe_keys.pth')

        self.db_dir = db_dir
        print(f"[CIRService] 数据库已加载: {len(self.db_image_names)} 张图像")

    def initialize(self,
                   architecture: str = 'resnet101',
                   pooling: str = 'gem',
                   whitening: bool = True,
                   model_path: Optional[str] = None,
                   index_dir: Optional[str] = None):
        """
        初始化服务（兼容旧 API）。

        Args:
            architecture: 骨干网络架构 (resnet50, resnet101, resnet152 等)
            pooling: 池化方法 (mac, gem, spoc, rmac)
            whitening: 是否使用白化
            model_path: 预训练模型路径（可选，不提供则使用 torchvision 预训练）
            index_dir: 索引目录
        """
        if self._initialized:
            return

        # 如果提供了模型路径，加载完整模型
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            # 自动初始化 torchvision 预训练模型
            print(f"[CIRService] 自动初始化预训练模型: {architecture}-{pooling}")
            self._init_pretrained_model(architecture, pooling, whitening)

        if index_dir and os.path.exists(index_dir):
            self.load_database(index_dir)

        self._initialized = True

    def _init_pretrained_model(self,
                                architecture: str = 'resnet101',
                                pooling: str = 'gem',
                                whitening: bool = True):
        """
        使用 torchvision 预训练权重初始化模型。

        Args:
            architecture: 骨干网络架构
            pooling: 池化方法
            whitening: 是否使用白化
        """
        net_params = {
            'architecture': architecture,
            'pooling': pooling,
            'local_whitening': False,
            'regional': False,
            'whitening': whitening,
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'pretrained': True
        }

        # 初始化网络
        self.model = init_network(net_params)
        self.model.to(self.device)
        self.model.eval()

        # 保存元信息
        self.model_meta = self.model.meta
        self.feature_dim = self.model.meta['outputdim']

        # 更新 ASPE 实例的特征维度
        self.aspe = ASPEForCNN(feature_dim=self.feature_dim, device=str(self.device))

        print(f"[CIRService] 预训练模型初始化成功: {architecture}-{pooling}, 特征维度: {self.feature_dim}")

    def get_status(self) -> Dict[str, Any]:
        """获取服务状态。"""
        return {
            "initialized": self._initialized,
            "model_loaded": self.model is not None,
            "indexed": self.db_features is not None,
            "index_size": len(self.db_image_names) if self.db_image_names else 0,
            "feature_dim": self.feature_dim,
            "keys_loaded": self.aspe.M1 is not None
        }

    def get_index_size(self) -> int:
        """获取索引图像数量。"""
        return len(self.db_image_names) if self.db_image_names else 0

    def get_feature_dim(self) -> int:
        """获取特征维度。"""
        return self.feature_dim

    @property
    def is_indexed(self) -> bool:
        """检查是否已构建索引。"""
        return self.db_features is not None


# Singleton accessor
def get_cir_service() -> CIRService:
    """Get the CIR service singleton."""
    return CIRService.get_instance()