"""
哈希码缓存服务：预计算和缓存数据库哈希码

缓存结构：
backend/cache/
├── dcmh/                    # DCMH 缓存目录
│   ├── flickr25k/           # 数据集目录
│   │   ├── database.npz     # 数据库哈希码
│   │   ├── query.npz        # 查询哈希码
│   │   └── encrypted.npz    # 加密数据库
│   └── mscoco/
│       └── ...
└── cir/                     # CIR 缓存目录
    ├── roxford5k/
    │   ├── features.npz
    │   ├── query_features.npz
    │   └── encrypted.npz
    └── rparis6k/
        └── ...

提供：
- 预计算数据库哈希码
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

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 缓存目录
DEFAULT_CACHE_DIR = PROJECT_ROOT / "backend" / "cache"
DCMH_CACHE_DIR = DEFAULT_CACHE_DIR / "dcmh"
CIR_CACHE_DIR = DEFAULT_CACHE_DIR / "cir"


class HashCacheService:
    """
    哈希码缓存服务。

    提供：
    - 预计算数据库哈希码
    - 缓存 ASPE 加密结果
    - 快速加载缓存
    - 支持多数据集（DCMH 和 CIR）
    """

    # 支持的 DCMH 数据集
    DCMH_DATASETS = ['flickr25k', 'mscoco', 'iapr_tc12']
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
        self.database_labels: Optional[np.ndarray] = None
        self.encrypted_database: Optional[np.ndarray] = None
        self.query_codes: Optional[np.ndarray] = None
        self.query_labels: Optional[np.ndarray] = None

        # 当前数据集
        self._current_dataset: Optional[str] = None

        # 缓存元数据
        self._metadata: Dict[str, Any] = {}

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
            dataset: 数据集名称（flickr25k, mscoco 等）
            name: 缓存类型（database, query, encrypted）

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

        print(f"已保存 DCMH 缓存：{cache_path}")

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

        print(f"已加载 DCMH 缓存：{cache_path}")
        return codes, labels

    def save_dcmh_encrypted(self, encrypted_database: np.ndarray, dataset: str = "flickr25k"):
        """保存 DCMH 加密数据库缓存。"""
        cache_path = self.get_dcmh_cache_path(dataset, "encrypted")
        np.savez(cache_path, encrypted=encrypted_database)
        print(f"已保存 DCMH 加密缓存：{cache_path}")

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

        print(f"已保存缓存：{cache_path}")

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
            print(f"缓存不存在：{cache_path}")
            return None, None

        data = np.load(cache_path)
        codes = data.get('codes')
        labels = data.get('labels')

        print(f"已加载缓存：{cache_path}")
        return codes, labels

    def save_encrypted_cache(self, encrypted_database: np.ndarray):
        """保存加密的数据库缓存（兼容旧版本）。"""
        cache_path = self.get_cache_path("encrypted_database")
        np.savez(cache_path, encrypted=encrypted_database)
        print(f"已保存加密缓存：{cache_path}")

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
        构建数据库哈希码缓存。

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
        txt_dataset = dataset_service.create_text_dataset(retrieval_indices)

        # 生成哈希码
        print("正在生成数据库哈希码...")
        all_codes = []
        all_labels = []

        with torch.no_grad():
            for batch_idx, (images, _) in enumerate(img_loader):
                codes = dcmh_service.generate_image_code(images)
                all_codes.append(codes.cpu().numpy())

                if batch_idx % 50 == 0:
                    print(f"  处理进度：{batch_idx * batch_size} / {len(retrieval_indices)}")

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

        print(f"数据库哈希码生成完成：{database_codes.shape}")
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
        _, query_indices, _ = dataset_service.get_data_split_indices()  # (train, query, retrieval)

        # 创建 DataLoader
        img_loader = dataset_service.create_image_dataloader(
            query_indices, batch_size=batch_size
        )

        # 生成哈希码
        print("正在生成查询集哈希码...")
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

        print(f"查询集哈希码生成完成：{query_codes.shape}")
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
        print("正在加密数据库...")
        encrypted = aspe_service.encrypt_database(
            self.database_codes, self.database_labels
        )

        # 保存缓存（新格式）
        self.save_dcmh_encrypted(encrypted, dataset)

        self.encrypted_database = encrypted
        print(f"数据库加密完成：{encrypted.shape}")

        return encrypted

    def clear_cache(self):
        """清除所有缓存。"""
        import shutil
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        print("缓存已清除")

    def get_cache_info(self) -> Dict[str, Any]:
        """
        获取缓存信息（新版，支持多数据集）。

        返回：
            包含 DCMH 和 CIR 各数据集缓存状态的字典
        """
        dcmh_info = {}
        for dataset in self.DCMH_DATASETS:
            codes, _ = self.load_dcmh_cache(dataset, "database")
            dcmh_info[dataset] = {
                "database_cached": self.dcmh_cache_exists(dataset, "database"),
                "query_cached": self.dcmh_cache_exists(dataset, "query"),
                "encrypted_cached": self.dcmh_cache_exists(dataset, "encrypted"),
                "database_size": codes.shape[0] if codes is not None else 0,
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
        print(f"已保存 CIR 特征缓存：{cache_path}")

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

        print(f"已加载 CIR 特征缓存：{cache_path}")
        return features, image_names

    def save_cir_encrypted(self, encrypted_features: np.ndarray, dataset: str):
        """保存 CIR 加密特征缓存。"""
        cache_path = self.get_cir_cache_path(dataset, "encrypted")
        np.savez(cache_path, encrypted=encrypted_features)
        print(f"已保存 CIR 加密缓存：{cache_path}")

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
                       force_rebuild: bool = False) -> Tuple[np.ndarray, List[str]]:
        """
        构建 CIR 特征缓存。

        参数：
            cir_service: CIR 服务实例
            image_dir: 图像目录
            dataset: 数据集名称
            force_rebuild: 是否强制重建

        返回：
            (features, image_names) 元组
        """
        # 检查缓存
        if not force_rebuild:
            features, image_names = self.load_cir_features(dataset)
            if features is not None:
                cir_service.db_features = torch.from_numpy(features)
                cir_service.db_image_names = image_names
                return features, image_names

        # 构建数据库
        features, image_names = cir_service.build_database(image_dir)

        # 保存缓存
        if isinstance(features, torch.Tensor):
            features = features.numpy()
        self.save_cir_features(features, image_names, dataset)

        return features, image_names

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