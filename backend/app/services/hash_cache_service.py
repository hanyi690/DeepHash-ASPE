"""
哈希码缓存服务：预计算和缓存数据库哈希码

缓存结构：
backend/cache/
├── dcmh/                    # DCMH 缓存目录
│   ├── flickr25k/           # 数据集目录
│   │   ├── labels.npz             # 标签（只存一次，优化空间）
│   │   ├── database_image.npz     # 图像哈希码（用于文本→图像检索）
│   │   ├── database_text.npz      # 文本哈希码（用于图像→文本检索）
│   │   ├── query.npz              # 查询哈希码
│   │   └── encrypted.npz          # 加密数据库
│   └── nuswide/
│       └── ...
└── cir/                     # CIR 缓存目录
    ├── roxford5k/
    │   ├── features.npz           # 明文特征
    │   ├── encrypted.npz          # 加密特征
    │   └── image_names.npz
    └── rparis6k/
        └── ...

优化说明：
- labels 单独存储，避免在 image/text 缓存中重复（节省约 380MB）
- 兼容旧格式：自动检测并加载旧格式缓存

提供：
- 预计算数据库哈希码（图像和文本）
- 缓存加密检索库
- 快速加载缓存
- 支持多数据集
"""

import numpy as np
import torch
from typing import Optional, Dict, Any, Tuple, List
from pathlib import Path
import json
import sys
import logging

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)

# 缓存目录
DEFAULT_CACHE_DIR = PROJECT_ROOT / "backend" / "cache"
DCMH_CACHE_DIR = DEFAULT_CACHE_DIR / "dcmh"
CIR_CACHE_DIR = DEFAULT_CACHE_DIR / "cir"


class HashCacheService:
    """
    哈希码缓存服务。

    提供：
    - 预计算数据库哈希码（图像和文本）
    - 缓存 ASPE 加密结果
    - 快速加载缓存
    - 支持多数据集（DCMH 和 CIR）
    """

    # 支持的 DCMH 数据集
    DCMH_DATASETS = ['flickr25k',  'iapr_tc12', 'nuswide']
    # 支持的 CIR 数据集
    CIR_DATASETS = ['roxford5k', 'rparis6k', 'oxford5k', 'paris6k']

    def __init__(self,
                 cache_dir: Optional[str] = None,
                 bit_dim: int = 64):
        """
        初始化缓存服务。

        参数：
            cache_dir: 缓存目录（兼容旧版本，默认使用 backend/cache）
            bit_dim: 哈希码位数
        """
        self.cache_dir = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
        self.bit_dim = bit_dim

        # 确保缓存目录存在
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        DCMH_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        CIR_CACHE_DIR.mkdir(parents=True, exist_ok=True)

        # 缓存数据（内存中的当前数据集）
        self.database_codes: Optional[np.ndarray] = None
        self.database_text_codes: Optional[np.ndarray] = None  # 文本哈希码
        self.database_labels: Optional[np.ndarray] = None
        self.encrypted_database: Optional[np.ndarray] = None
        self.query_codes: Optional[np.ndarray] = None
        self.query_labels: Optional[np.ndarray] = None

        # 当前数据集
        self._current_dataset: Optional[str] = None

        # 缓存元数据
        self._metadata: Dict[str, Any] = {}

        logger.info(f"[HashCacheService] 初始化：cache_dir={self.cache_dir}")

    # ==================== 通用缓存方法 ====================

    def get_cache_path(self, name: str) -> Path:
        """获取缓存文件路径（兼容旧版本）。"""
        return self.cache_dir / f"{name}.npz"

    def get_metadata_path(self) -> Path:
        """获取元数据文件路径。"""
        return self.cache_dir / "metadata.json"

    def exists(self, name: str = "database") -> bool:
        """检查缓存是否存在（兼容旧版本）。"""
        return self.get_cache_path(name).exists()

    # ==================== DCMH 数据集缓存方法 ====================

    def get_dcmh_cache_path(self, dataset: str, name: str = "database") -> Path:
        """
        获取 DCMH 数据集缓存文件路径。

        参数：
            dataset: 数据集名称（flickr25k, nuswide 等）
            name: 缓存类型
                - database_image: 图像哈希码
                - database_text: 文本哈希码
                - database: 兼容旧版（图像哈希码）
                - query: 查询哈希码
                - encrypted: 加密哈希码

        返回：
            缓存文件路径
        """
        cache_dir = DCMH_CACHE_DIR / dataset
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir / f"{name}.npz"

    def save_dcmh_cache(self,
                        codes: np.ndarray,
                        labels: Optional[np.ndarray] = None,
                        dataset: str = "flickr25k",
                        name: str = "database"):
        """
        保存 DCMH 哈希码缓存。

        参数：
            codes: 哈希码数组
            labels: 标签数组（可选）
            dataset: 数据集名称
            name: 缓存类型
        """
        cache_path = self.get_dcmh_cache_path(dataset, name)

        if labels is not None:
            np.savez(cache_path, codes=codes, labels=labels)
        else:
            np.savez(cache_path, codes=codes)

        logger.info(f"[HashCacheService] 已保存 DCMH 缓存：{cache_path}")

    def load_dcmh_cache(self,
                        dataset: str = "flickr25k",
                        name: str = "database") -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        加载 DCMH 哈希码缓存。

        参数：
            dataset: 数据集名称
            name: 缓存类型

        返回：
            (codes, labels) 元组
        """
        cache_path = self.get_dcmh_cache_path(dataset, name)

        if not cache_path.exists():
            return None, None

        data = np.load(cache_path)
        codes = data.get('codes')
        labels = data.get('labels')

        logger.info(f"[HashCacheService] 已加载 DCMH 缓存：{cache_path}")
        return codes, labels

    def save_dcmh_labels(self, labels: np.ndarray, dataset: str = "flickr25k"):
        """
        单独保存 DCMH 标签缓存。

        优化：labels 只存储一次，避免在 image/text 缓存中重复。

        参数：
            labels: 标签数组
            dataset: 数据集名称
        """
        cache_path = self.get_dcmh_cache_path(dataset, "labels")
        np.savez(cache_path, labels=labels)
        logger.info(f"[HashCacheService] 已保存 DCMH 标签缓存：{cache_path}")

    def load_dcmh_labels(self, dataset: str = "flickr25k") -> Optional[np.ndarray]:
        """
        单独加载 DCMH 标签缓存。

        优先从 labels.npz 加载，如果不存在则尝试从旧格式缓存中提取。

        参数：
            dataset: 数据集名称

        返回：
            labels 数组，如果不存在则返回 None
        """
        # 优先从 labels.npz 加载
        cache_path = self.get_dcmh_cache_path(dataset, "labels")
        if cache_path.exists():
            data = np.load(cache_path)
            labels = data.get('labels')
            logger.info(f"[HashCacheService] 已加载 DCMH 标签缓存：{cache_path}")
            return labels

        # 兼容旧格式：从 database_image.npz 或 database.npz 中提取
        for name in ["database_image", "database"]:
            path = self.get_dcmh_cache_path(dataset, name)
            if path.exists():
                data = np.load(path)
                if 'labels' in data:
                    labels = data.get('labels')
                    logger.info(f"[HashCacheService] 从旧格式缓存提取标签：{path}")
                    return labels

        return None

    def save_dcmh_encrypted(self, encrypted_database: np.ndarray, dataset: str = "flickr25k"):
        """保存 DCMH 加密数据库缓存。"""
        cache_path = self.get_dcmh_cache_path(dataset, "encrypted")
        np.savez(cache_path, encrypted=encrypted_database)
        logger.info(f"[HashCacheService] 已保存 DCMH 加密缓存：{cache_path}")

    def load_dcmh_encrypted(self, dataset: str = "flickr25k") -> Optional[np.ndarray]:
        """加载 DCMH 加密数据库缓存。"""
        cache_path = self.get_dcmh_cache_path(dataset, "encrypted")

        if not cache_path.exists():
            return None

        data = np.load(cache_path)
        return data.get('encrypted')

    def dcmh_cache_exists(self, dataset: str, name: str = "database") -> bool:
        """检查 DCMH 缓存是否存在。"""
        return self.get_dcmh_cache_path(dataset, name).exists()

    # ==================== 完整数据库缓存构建 ====================

    def build_full_database_cache(self,
                                   dcmh_service,
                                   dataset_service,
                                   batch_size: int = 64,
                                   force_rebuild: bool = False,
                                   dataset: str = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        构建完整的数据库缓存（图像哈希码 + 文本哈希码 + 标签均值）。

        参数：
            dcmh_service: DCMH 服务实例
            dataset_service: 数据集服务实例
            batch_size: 批次大小
            force_rebuild: 是否强制重建
            dataset: 数据集名称

        返回：
            (image_codes, text_codes, labels) 元组
        """
        # 确定数据集名称
        if dataset is None:
            dataset = getattr(dataset_service, 'dataset_name', 'flickr25k')

        logger.info(f"[HashCacheService] 构建完整数据库缓存：{dataset}")

        # 检查是否需要构建
        if not force_rebuild:
            image_codes, _ = self.load_dcmh_cache(dataset, "database_image")
            text_codes, _ = self.load_dcmh_cache(dataset, "database_text")
            labels = self.load_dcmh_labels(dataset)

            if image_codes is not None and text_codes is not None and labels is not None:
                logger.info(f"[HashCacheService] 缓存已存在，跳过构建")
                self.database_codes = image_codes
                self.database_text_codes = text_codes
                self.database_labels = labels
                return image_codes, text_codes, labels

        # 加载数据
        dataset_service.load_data()
        train_indices, _, retrieval_indices = dataset_service.get_data_split_indices()

        # ========== 1. 生成图像哈希码 ==========
        logger.info("[HashCacheService] 正在生成图像哈希码...")
        img_loader = dataset_service.create_image_dataloader(
            retrieval_indices, batch_size=batch_size
        )

        all_image_codes = []
        with torch.no_grad():
            for batch_idx, (images, _) in enumerate(img_loader):
                codes = dcmh_service.generate_image_code(images)
                all_image_codes.append(codes.cpu().numpy())

                if batch_idx % 50 == 0:
                    logger.info(f"  图像哈希码进度：{batch_idx * batch_size} / {len(retrieval_indices)}")

        database_image_codes = np.vstack(all_image_codes)

        # ========== 2. 生成文本哈希码 ==========
        logger.info("[HashCacheService] 正在生成文本哈希码...")
        txt_dataset = dataset_service.create_text_dataset(retrieval_indices)
        txt_loader = torch.utils.data.DataLoader(txt_dataset, batch_size=batch_size)

        all_text_codes = []
        with torch.no_grad():
            for batch_idx, (tags, labels, indices) in enumerate(txt_loader):
                codes = dcmh_service.generate_text_code(tags)
                all_text_codes.append(codes.cpu().numpy())

                if batch_idx % 50 == 0:
                    logger.info(f"  文本哈希码进度：{batch_idx * batch_size} / {len(retrieval_indices)}")

        database_text_codes = np.vstack(all_text_codes)

        # ========== 3. 获取标签（YAll） ==========
        database_labels = dataset_service.get_yall(retrieval_indices)

        # ========== 4. 保存缓存 ==========
        # 优化：labels 单独存储，避免重复
        self.save_dcmh_labels(database_labels, dataset)
        # 图像哈希码只存储 codes
        self.save_dcmh_cache(database_image_codes, None, dataset, "database_image")
        # 文本哈希码只存储 codes
        self.save_dcmh_cache(database_text_codes, None, dataset, "database_text")

        # 更新内存缓存
        self.database_codes = database_image_codes
        self.database_text_codes = database_text_codes
        self.database_labels = database_labels
        self._current_dataset = dataset

        logger.info(f"[HashCacheService] 数据库缓存构建完成："
                   f"图像 {database_image_codes.shape}, 文本 {database_text_codes.shape}")

        return database_image_codes, database_text_codes, database_labels

    def build_from_mat(self,
                       dcmh_service,
                       mat_path: str = None,
                       dataset: str = 'flickr25k',
                       batch_size: int = 64,
                       force_rebuild: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        从 FLICKR-25K.mat 构建缓存。

        这个方法直接从 .mat 文件读取数据，确保与训练时使用相同的数据源。

        参数：
            dcmh_service: DCMH 服务实例
            mat_path: .mat 文件路径
            dataset: 数据集名称
            batch_size: 批次大小
            force_rebuild: 是否强制重建

        返回：
            (image_codes, text_codes, labels) 元组
        """
        import h5py

        # 默认路径
        if mat_path is None:
            mat_path = str(PROJECT_ROOT / 'data' / 'FLICKR-25K.mat')

        logger.info(f"[HashCacheService] 从 .mat 构建缓存: {mat_path}")

        # 检查缓存
        if not force_rebuild:
            image_codes, _ = self.load_dcmh_cache(dataset, "database_image")
            text_codes, _ = self.load_dcmh_cache(dataset, "database_text")
            labels = self.load_dcmh_labels(dataset)

            if image_codes is not None and text_codes is not None and labels is not None:
                logger.info(f"[HashCacheService] 缓存已存在，跳过构建")
                self.database_codes = image_codes
                self.database_text_codes = text_codes
                self.database_labels = labels
                return image_codes, text_codes, labels

        # 读取 .mat 文件
        with h5py.File(mat_path, 'r') as f:
            logger.info(f"[HashCacheService] 读取 .mat 文件...")
            logger.info(f"  可用数据集: {list(f.keys())}")

            # Flickr25K 数据划分：查询集 0-1999, 训练集 2000-11999, 检索库 2000-20015
            retrieval_start = 2000
            retrieval_end = 20015

            # 读取图像数据
            images = f['images'][retrieval_start:retrieval_end]
            logger.info(f"  图像数据: {images.shape}")

            # 读取文本标签 YAll (1386 维)
            yall = f['YAll'][retrieval_start:retrieval_end]
            yall = np.array(yall.toarray() if hasattr(yall, 'toarray') else yall)
            logger.info(f"  YAll 标签: {yall.shape}")

            # 读取类别标签 LAll (24 维) - 如果需要
            if 'LAll' in f:
                lall = f['LAll'][retrieval_start:retrieval_end]
                lall = np.array(lall.toarray() if hasattr(lall, 'toarray') else lall)
            else:
                lall = None

        num_samples = images.shape[0]
        logger.info(f"  检索库样本数: {num_samples}")

        # ========== 1. 生成图像哈希码 ==========
        logger.info("[HashCacheService] 正在生成图像哈希码...")
        all_image_codes = []

        with torch.no_grad():
            for i in range(0, num_samples, batch_size):
                batch_images = images[i:i + batch_size]
                # 转换为 tensor，添加通道维度 (如果需要)
                batch_tensor = torch.from_numpy(batch_images).float()
                if batch_tensor.dim() == 3:
                    # [N, H, W] -> [N, C, H, W]
                    batch_tensor = batch_tensor.unsqueeze(1)
                    # 扩展到 3 通道
                    batch_tensor = batch_tensor.repeat(1, 3, 1, 1)

                codes = dcmh_service.generate_image_code(batch_tensor)
                all_image_codes.append(codes.cpu().numpy())

                if i % (batch_size * 20) == 0:
                    logger.info(f"  图像哈希码进度: {i} / {num_samples}")

        database_image_codes = np.vstack(all_image_codes)
        logger.info(f"  图像哈希码完成: {database_image_codes.shape}")

        # ========== 2. 生成文本哈希码 ==========
        logger.info("[HashCacheService] 正在生成文本哈希码...")
        all_text_codes = []

        with torch.no_grad():
            for i in range(0, num_samples, batch_size):
                batch_yall = yall[i:i + batch_size]
                batch_tensor = torch.from_numpy(batch_yall).float()

                codes = dcmh_service.generate_text_code(batch_tensor)
                all_text_codes.append(codes.cpu().numpy())

        database_text_codes = np.vstack(all_text_codes)
        logger.info(f"  文本哈希码完成: {database_text_codes.shape}")

        # ========== 3. 保存缓存 ==========
        # 优化：labels 单独存储，避免重复
        self.save_dcmh_labels(yall, dataset)
        # 图像哈希码只存储 codes
        self.save_dcmh_cache(database_image_codes, None, dataset, "database_image")
        # 文本哈希码只存储 codes
        self.save_dcmh_cache(database_text_codes, None, dataset, "database_text")

        # 更新内存缓存
        self.database_codes = database_image_codes
        self.database_text_codes = database_text_codes
        self.database_labels = yall
        self._current_dataset = dataset

        logger.info(f"[HashCacheService] .mat 缓存构建完成: "
                   f"图像 {database_image_codes.shape}, 文本 {database_text_codes.shape}")

        return database_image_codes, database_text_codes, yall

    def load_full_database(self,
                           dataset: str = "flickr25k") -> Tuple[Optional[np.ndarray],
                                                                  Optional[np.ndarray],
                                                                  Optional[np.ndarray]]:
        """
        加载完整的数据库缓存（图像哈希码 + 文本哈希码 + 标签）。

        优化：labels 从单独的 labels.npz 文件加载。

        参数：
            dataset: 数据集名称

        返回：
            (image_codes, text_codes, labels) 元组
        """
        # 加载哈希码
        image_codes, _ = self.load_dcmh_cache(dataset, "database_image")
        text_codes, _ = self.load_dcmh_cache(dataset, "database_text")

        # 加载标签（从分离的 labels.npz 或旧格式中）
        labels = self.load_dcmh_labels(dataset)

        # 兼容旧版：如果 image_codes 不存在，尝试从 database.npz 加载
        if image_codes is None:
            image_codes, labels = self.load_dcmh_cache(dataset, "database")

        return image_codes, text_codes, labels

    # ==================== 兼容旧版本的方法 ====================

    def save_cache(self,
                  codes: np.ndarray,
                  labels: Optional[np.ndarray] = None,
                  name: str = "database"):
        """
        保存哈希码缓存（兼容旧版本）。

        参数：
            codes: 哈希码数组
            labels: 标签数组（可选）
            name: 缓存名称
        """
        cache_path = self.get_cache_path(name)

        if labels is not None:
            np.savez(cache_path, codes=codes, labels=labels)
        else:
            np.savez(cache_path, codes=codes)

        logger.info(f"[HashCacheService] 已保存缓存：{cache_path}")

    def load_cache(self, name: str = "database") -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        加载哈希码缓存（兼容旧版本）。

        参数：
            name: 缓存名称

        返回：
            (codes, labels) 元组
        """
        cache_path = self.get_cache_path(name)

        if not cache_path.exists():
            logger.warning(f"[HashCacheService] 缓存不存在：{cache_path}")
            return None, None

        data = np.load(cache_path)
        codes = data.get('codes')
        labels = data.get('labels')

        logger.info(f"[HashCacheService] 已加载缓存：{cache_path}")
        return codes, labels

    def save_encrypted_cache(self, encrypted_database: np.ndarray):
        """保存加密的数据库缓存（兼容旧版本）。"""
        cache_path = self.get_cache_path("encrypted_database")
        np.savez(cache_path, encrypted=encrypted_database)
        logger.info(f"[HashCacheService] 已保存加密缓存：{cache_path}")

    def load_encrypted_cache(self) -> Optional[np.ndarray]:
        """加载加密的数据库缓存（兼容旧版本）。"""
        cache_path = self.get_cache_path("encrypted_database")

        if not cache_path.exists():
            return None

        data = np.load(cache_path)
        return data.get('encrypted')

    def build_database_cache(self,
                            dcmh_service,
                            dataset_service,
                            batch_size: int = 64,
                            force_rebuild: bool = False,
                            dataset: str = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        构建数据库哈希码缓存（兼容旧版本，仅生成图像哈希码）。

        注意：建议使用 build_full_database_cache() 以获得完整功能。

        参数：
            dcmh_service: DCMH 服务实例
            dataset_service: 数据集服务实例
            batch_size: 批次大小
            force_rebuild: 是否强制重建
            dataset: 数据集名称（如 flickr25k，可选）

        返回：
            (database_codes, database_labels) 元组
        """
        # 确定数据集名称
        if dataset is None:
            dataset = getattr(dataset_service, 'dataset_name', 'flickr25k')

        # 检查新格式缓存
        if not force_rebuild and self.dcmh_cache_exists(dataset, "database"):
            codes, labels = self.load_dcmh_cache(dataset, "database")
            if codes is not None:
                self.database_codes = codes
                self.database_labels = labels
                self._current_dataset = dataset
                return codes, labels

        # 检查旧格式缓存（兼容）
        if not force_rebuild and self.exists("database"):
            codes, labels = self.load_cache("database")
            if codes is not None:
                self.database_codes = codes
                self.database_labels = labels
                # 迁移到新格式
                self.save_dcmh_cache(codes, labels, dataset, "database")
                return codes, labels

        # 加载数据
        dataset_service.load_data()
        _, _, retrieval_indices = dataset_service.get_data_split_indices()

        # 创建 DataLoader
        img_loader = dataset_service.create_image_dataloader(
            retrieval_indices, batch_size=batch_size
        )

        # 生成哈希码
        logger.info("[HashCacheService] 正在生成数据库哈希码...")
        all_codes = []

        with torch.no_grad():
            for batch_idx, (images, _) in enumerate(img_loader):
                codes = dcmh_service.generate_image_code(images)
                all_codes.append(codes.cpu().numpy())

                if batch_idx % 50 == 0:
                    logger.info(f"  处理进度：{batch_idx * batch_size} / {len(retrieval_indices)}")

        # 获取标签
        all_labels = dataset_service.get_labels(retrieval_indices)

        # 合并
        database_codes = np.vstack(all_codes)
        database_labels = all_labels

        # 保存缓存（新格式）
        self.save_dcmh_cache(database_codes, database_labels, dataset, "database")

        self.database_codes = database_codes
        self.database_labels = database_labels
        self._current_dataset = dataset

        logger.info(f"[HashCacheService] 数据库哈希码生成完成：{database_codes.shape}")
        return database_codes, database_labels

    def build_query_cache(self,
                         dcmh_service,
                         dataset_service,
                         batch_size: int = 64,
                         force_rebuild: bool = False,
                         dataset: str = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        构建查询集哈希码缓存。

        参数：
            dcmh_service: DCMH 服务实例
            dataset_service: 数据集服务实例
            batch_size: 批次大小
            force_rebuild: 是否强制重建
            dataset: 数据集名称（可选）

        返回：
            (query_codes, query_labels) 元组
        """
        # 确定数据集名称
        if dataset is None:
            dataset = getattr(dataset_service, 'dataset_name', 'flickr25k')

        # 检查新格式缓存
        if not force_rebuild and self.dcmh_cache_exists(dataset, "query"):
            codes, labels = self.load_dcmh_cache(dataset, "query")
            if codes is not None:
                self.query_codes = codes
                self.query_labels = labels
                return codes, labels

        # 检查旧格式缓存（兼容）
        if not force_rebuild and self.exists("query"):
            codes, labels = self.load_cache("query")
            if codes is not None:
                self.query_codes = codes
                self.query_labels = labels
                # 迁移到新格式
                self.save_dcmh_cache(codes, labels, dataset, "query")
                return codes, labels

        # 加载数据
        dataset_service.load_data()
        _, query_indices, _ = dataset_service.get_data_split_indices()

        # 创建 DataLoader
        img_loader = dataset_service.create_image_dataloader(
            query_indices, batch_size=batch_size
        )

        # 生成哈希码
        logger.info("[HashCacheService] 正在生成查询集哈希码...")
        all_codes = []

        with torch.no_grad():
            for images, _ in img_loader:
                codes = dcmh_service.generate_image_code(images)
                all_codes.append(codes.cpu().numpy())

        # 获取标签
        all_labels = dataset_service.get_labels(query_indices)

        # 合并
        query_codes = np.vstack(all_codes)
        query_labels = all_labels

        # 保存缓存（新格式）
        self.save_dcmh_cache(query_codes, query_labels, dataset, "query")

        self.query_codes = query_codes
        self.query_labels = query_labels

        logger.info(f"[HashCacheService] 查询集哈希码生成完成：{query_codes.shape}")
        return query_codes, query_labels

    def build_encrypted_cache(self,
                             aspe_service,
                             force_rebuild: bool = False,
                             dataset: str = None) -> np.ndarray:
        """
        构建加密数据库缓存。

        参数：
            aspe_service: ASPE 服务实例
            force_rebuild: 是否强制重建
            dataset: 数据集名称（可选，默认使用当前数据集）

        返回：
            加密的数据库
        """
        # 确定数据集名称
        if dataset is None:
            dataset = self._current_dataset or 'flickr25k'

        # 检查新格式缓存
        if not force_rebuild:
            encrypted = self.load_dcmh_encrypted(dataset)
            if encrypted is not None:
                self.encrypted_database = encrypted
                aspe_service.encrypted_database = encrypted
                return encrypted

        # 检查旧格式缓存（兼容）
        if not force_rebuild:
            encrypted = self.load_encrypted_cache()
            if encrypted is not None:
                self.encrypted_database = encrypted
                aspe_service.encrypted_database = encrypted
                # 迁移到新格式
                self.save_dcmh_encrypted(encrypted, dataset)
                return encrypted

        # 检查明文数据库
        if self.database_codes is None:
            raise ValueError("请先构建数据库哈希码缓存")

        # 加密
        logger.info("[HashCacheService] 正在加密数据库...")
        encrypted = aspe_service.encrypt_database(
            self.database_codes, self.database_labels
        )

        # 保存缓存（新格式）
        self.save_dcmh_encrypted(encrypted, dataset)

        self.encrypted_database = encrypted
        logger.info(f"[HashCacheService] 数据库加密完成：{encrypted.shape}")

        return encrypted

    def clear_cache(self):
        """清除所有缓存。"""
        import shutil
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info("[HashCacheService] 缓存已清除")

    def get_cache_info(self) -> Dict[str, Any]:
        """
        获取缓存信息（新版，支持多数据集）。

        返回：
            包含 DCMH 和 CIR 各数据集缓存状态的字典
        """
        dcmh_info = {}
        for dataset in self.DCMH_DATASETS:
            codes, _ = self.load_dcmh_cache(dataset, "database")
            image_codes, _ = self.load_dcmh_cache(dataset, "database_image")
            text_codes, _ = self.load_dcmh_cache(dataset, "database_text")
            labels = self.load_dcmh_labels(dataset)

            dcmh_info[dataset] = {
                "database_cached": self.dcmh_cache_exists(dataset, "database"),
                "database_image_cached": self.dcmh_cache_exists(dataset, "database_image"),
                "database_text_cached": self.dcmh_cache_exists(dataset, "database_text"),
                "labels_cached": self.dcmh_cache_exists(dataset, "labels"),
                "query_cached": self.dcmh_cache_exists(dataset, "query"),
                "encrypted_cached": self.dcmh_cache_exists(dataset, "encrypted"),
                "database_size": codes.shape[0] if codes is not None else (image_codes.shape[0] if image_codes is not None else 0),
                "bit_dim": self.bit_dim
            }

        return {
            "cache_dir": str(self.cache_dir),
            "dcmh": dcmh_info,
            "cir": self.get_cir_cache_info()["datasets"]
        }

    def get_cache_info_legacy(self) -> Dict[str, Any]:
        """获取缓存信息（兼容旧版本）。"""
        return {
            "cache_dir": str(self.cache_dir),
            "bit_dim": self.bit_dim,
            "database_cached": self.exists("database"),
            "query_cached": self.exists("query"),
            "encrypted_cached": self.exists("encrypted_database"),
            "database_size": self.database_codes.shape[0] if self.database_codes is not None else 0,
            "query_size": self.query_codes.shape[0] if self.query_codes is not None else 0
        }

    # ==================== CIR 特征缓存方法 ====================

    def get_cir_cache_path(self, dataset: str, name: str = "features") -> Path:
        """
        获取 CIR 缓存文件路径。

        参数：
            dataset: 数据集名称（如 roxford5k, rparis6k）
            name: 缓存类型（features, encrypted）

        返回：
            缓存文件路径
        """
        cir_cache_dir = CIR_CACHE_DIR / dataset
        cir_cache_dir.mkdir(parents=True, exist_ok=True)
        return cir_cache_dir / f"{name}.npz"

    def save_cir_features(self,
                         features: np.ndarray,
                         image_names: List[str],
                         dataset: str,
                         name: str = "features"):
        """
        保存 CIR 特征缓存。

        参数：
            features: 特征矩阵 [N, D]
            image_names: 图像名称列表
            dataset: 数据集名称
            name: 缓存类型
        """
        cache_path = self.get_cir_cache_path(dataset, name)
        np.savez(cache_path, features=features, image_names=np.array(image_names))
        logger.info(f"[HashCacheService] 已保存 CIR 特征缓存：{cache_path}")

    def load_cir_features(self, dataset: str, name: str = "features") -> Tuple[Optional[np.ndarray], Optional[List[str]]]:
        """
        加载 CIR 特征缓存。

        参数：
            dataset: 数据集名称
            name: 缓存类型

        返回：
            (features, image_names) 元组
        """
        cache_path = self.get_cir_cache_path(dataset, name)

        if not cache_path.exists():
            return None, None

        data = np.load(cache_path, allow_pickle=True)
        features = data.get('features')
        image_names = data.get('image_names')

        if image_names is not None:
            image_names = image_names.tolist()

        logger.info(f"[HashCacheService] 已加载 CIR 特征缓存：{cache_path}")
        return features, image_names

    def save_cir_encrypted(self, encrypted_features: np.ndarray, dataset: str):
        """保存 CIR 加密特征缓存。"""
        cache_path = self.get_cir_cache_path(dataset, "encrypted")
        np.savez(cache_path, encrypted=encrypted_features)
        logger.info(f"[HashCacheService] 已保存 CIR 加密缓存：{cache_path}")

    def load_cir_encrypted(self, dataset: str) -> Optional[np.ndarray]:
        """加载 CIR 加密特征缓存。"""
        cache_path = self.get_cir_cache_path(dataset, "encrypted")

        if not cache_path.exists():
            return None

        data = np.load(cache_path)
        return data.get('encrypted')

    def cir_cache_exists(self, dataset: str, name: str = "features") -> bool:
        """检查 CIR 缓存是否存在。"""
        return self.get_cir_cache_path(dataset, name).exists()

    def build_cir_cache(self,
                       cir_service,
                       image_dir: str,
                       dataset: str,
                       force_rebuild: bool = False,
                       skip_encryption: bool = False) -> Tuple[np.ndarray, List[str]]:
        """
        构建 CIR 特征缓存。

        参数：
            cir_service: CIR 服务实例
            image_dir: 图像目录
            dataset: 数据集名称
            force_rebuild: 是否强制重建
            skip_encryption: 是否跳过加密（仅保存明文特征）

        返回：
            (features, image_names) 元组 - 明文特征
        """
        # 检查明文特征缓存
        if not force_rebuild:
            features, image_names = self.load_cir_features(dataset)
            if features is not None:
                cir_service.db_plaintext_features = torch.from_numpy(features)
                cir_service.db_image_names = image_names
                return features, image_names

        # 收集图像文件
        import torch
        from pathlib import Path

        image_dir = Path(image_dir)
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_paths = [
            str(p) for p in image_dir.rglob('*')
            if p.suffix.lower() in image_extensions
        ]

        if not image_paths:
            raise ValueError(f"图像目录为空: {image_dir}")

        # 提取明文特征
        logger.info(f"[HashCacheService] 提取 {len(image_paths)} 张图像的特征...")
        features = cir_service.extract_features(image_paths)
        features_np = features.numpy().T  # [N, d]

        # 保存图像名称
        image_names = [Path(p).name for p in image_paths]

        # 保存明文特征缓存
        self.save_cir_features(features_np, image_names, dataset)

        # 更新服务状态
        cir_service.db_plaintext_features = torch.from_numpy(features_np)
        cir_service.db_image_names = image_names

        # 如果不跳过加密，生成加密特征
        if not skip_encryption:
            # 确保 ASPE 初始化并使用正确的特征维度
            actual_feature_dim = features_np.shape[1]
            if cir_service.aspe is None or cir_service.aspe.feature_dim != actual_feature_dim:
                from core.aspe.cnn_wrapper import ASPEForCNN
                cir_service.aspe = ASPEForCNN(feature_dim=actual_feature_dim, device=str(cir_service.device))
                cir_service.aspe.generate_keys()

            # 加密特征
            logger.info("[HashCacheService] 加密特征...")
            encrypted_features = cir_service.aspe.encrypt_database(features_np)
            cir_service.db_encrypted_features = encrypted_features

            # 保存加密缓存
            if isinstance(encrypted_features, torch.Tensor):
                encrypted_np = encrypted_features.cpu().numpy()
            else:
                encrypted_np = encrypted_features
            self.save_cir_encrypted(encrypted_np, dataset)

            # 保存密钥
            try:
                from backend.app.services.key_manager import get_key_manager
                key_manager = get_key_manager()
                key_manager._cir_keys = {
                    'M1': cir_service.aspe.M1.cpu(),
                    'M2': cir_service.aspe.M2.cpu(),
                    'S': cir_service.aspe.S.cpu(),
                    'M1_inv': cir_service.aspe.M1_inv.cpu() if cir_service.aspe.M1_inv is not None else None,
                    'M2_inv': cir_service.aspe.M2_inv.cpu() if cir_service.aspe.M2_inv is not None else None,
                    'feature_dim': actual_feature_dim
                }
                key_manager._save_cir_keys()
                logger.info("[HashCacheService] 已保存 CIR 密钥")
            except Exception as e:
                logger.warning(f"[HashCacheService] 保存密钥失败: {e}")

        return features_np, image_names

    def get_cir_cache_info(self) -> Dict[str, Any]:
        """获取 CIR 缓存信息。"""
        # 支持的数据集
        datasets = ['roxford5k', 'rparis6k']
        cache_info = {}

        for dataset in datasets:
            features, _ = self.load_cir_features(dataset)
            encrypted = self.load_cir_encrypted(dataset)

            cache_info[dataset] = {
                "features_cached": self.cir_cache_exists(dataset, "features"),
                "encrypted_cached": self.cir_cache_exists(dataset, "encrypted"),
                "database_size": features.shape[0] if features is not None else 0,
                "feature_dim": features.shape[1] if features is not None else 0
            }

        return {
            "cir_cache_dir": str(CIR_CACHE_DIR),
            "datasets": cache_info
        }


# 全局服务实例
_hash_cache_service: Optional[HashCacheService] = None


def get_hash_cache_service(**kwargs) -> HashCacheService:
    """获取或创建哈希缓存服务单例。"""
    global _hash_cache_service
    if _hash_cache_service is None:
        _hash_cache_service = HashCacheService(**kwargs)
    return _hash_cache_service