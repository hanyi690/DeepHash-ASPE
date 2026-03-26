"""
密钥管理器

负责管理所有加密密钥的生成、加载和保存。
确保服务重启后密钥一致性。

使用方法：
    from backend.app.services.key_manager import KeyManager

    # 获取单例实例
    km = KeyManager.get_instance()

    # 获取 DCMH 密钥 (Scheme 1)
    dcmh_keys = km.get_dcmh_keys(bit_dim=64)

    # 获取 DCMH Scheme 2 密钥
    dcmh_v2_keys = km.get_dcmh_v2_keys(bit_dim=64)

    # 获取 CIR 密钥
    cir_keys = km.get_cir_keys(feature_dim=2048)
"""

from pathlib import Path
import torch
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

# 密钥存储目录
KEYS_DIR = Path(__file__).parent.parent.parent / "backend" / "keys"


class KeyManager:
    """
    统一密钥管理器。

    负责管理 DCMH 和 CIR 的 ASPE 加密密钥。
    确保密钥持久化，服务重启后可复用加密缓存。

    密钥文件格式：
    - DCMH (Scheme 1): {M, M_inv, bit_dim, seed}
    - DCMH (Scheme 2): {M1, M2, S, M1_inv, M2_inv, bit_dim, d_prime, seed}
    - CIR (SkNN): {M1, M2, S, M1_inv, M2_inv, feature_dim}
    """

    _instance: Optional['KeyManager'] = None

    def __init__(self, keys_dir: Optional[Path] = None):
        """
        初始化密钥管理器。

        参数：
            keys_dir: 密钥存储目录（默认使用 backend/keys）
        """
        self.keys_dir = keys_dir or KEYS_DIR
        self.keys_dir.mkdir(parents=True, exist_ok=True)

        # 密钥文件路径
        self.dcmh_key_path = self.keys_dir / "dcmh_aspe_keys.pth"
        self.dcmh_v2_key_path = self.keys_dir / "dcmh_aspe_v2_keys.pth"
        self.cir_key_path = self.keys_dir / "cir_aspe_keys.pth"

        # 已加载的密钥缓存
        self._dcmh_keys: Optional[Dict[str, Any]] = None
        self._dcmh_v2_keys: Optional[Dict[str, Any]] = None
        self._cir_keys: Optional[Dict[str, Any]] = None

        logger.info(f"KeyManager 初始化：keys_dir={self.keys_dir}")

    @classmethod
    def get_instance(cls) -> 'KeyManager':
        """获取单例实例。"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls):
        """重置单例实例（用于测试）。"""
        cls._instance = None

    def get_dcmh_keys(self,
                      bit_dim: int = 64,
                      seed: int = 42,
                      force_generate: bool = False) -> Dict[str, Any]:
        """
        获取 DCMH ASPE 密钥。

        如果密钥文件存在，则加载；否则生成新密钥并保存。

        参数：
            bit_dim: 哈希码位数
            seed: 随机种子
            force_generate: 是否强制生成新密钥

        返回：
            包含 M, M_inv 的字典
        """
        if force_generate or self._dcmh_keys is None:
            if not force_generate and self.dcmh_key_path.exists():
                # 加载已有密钥
                self._dcmh_keys = self._load_dcmh_keys()
                logger.info(f"已加载 DCMH 密钥：{self.dcmh_key_path}")
            else:
                # 生成新密钥
                self._dcmh_keys = self._generate_dcmh_keys(bit_dim, seed)
                self._save_dcmh_keys()
                logger.info(f"已生成新 DCMH 密钥：{self.dcmh_key_path}")

        return self._dcmh_keys

    def get_cir_keys(self,
                     feature_dim: int = 2048,
                     force_generate: bool = False) -> Dict[str, Any]:
        """
        获取 CIR ASPE 密钥。

        如果密钥文件存在，则加载；否则生成新密钥并保存。

        参数：
            feature_dim: 特征维度
            force_generate: 是否强制生成新密钥

        返回：
            包含 M1, M2, S, M1_inv, M2_inv 的字典
        """
        if force_generate or self._cir_keys is None:
            if not force_generate and self.cir_key_path.exists():
                # 加载已有密钥
                self._cir_keys = self._load_cir_keys()
                logger.info(f"已加载 CIR 密钥：{self.cir_key_path}")
            else:
                # 生成新密钥
                self._cir_keys = self._generate_cir_keys(feature_dim)
                self._save_cir_keys()
                logger.info(f"已生成新 CIR 密钥：{self.cir_key_path}")

        return self._cir_keys

    def get_dcmh_v2_keys(self,
                         bit_dim: int = 64,
                         d_prime: Optional[int] = None,
                         seed: int = 42,
                         force_generate: bool = False) -> Dict[str, Any]:
        """
        获取 DCMH ASPE Scheme 2 密钥。

        如果密钥文件存在，则加载；否则生成新密钥并保存。

        参数：
            bit_dim: 哈希码位数
            d_prime: 扩展维度（默认：max(bit_dim+1, 80)）
            seed: 随机种子
            force_generate: 是否强制生成新密钥

        返回：
            包含 M1, M2, S, M1_inv, M2_inv 的字典
        """
        if force_generate or self._dcmh_v2_keys is None:
            if not force_generate and self.dcmh_v2_key_path.exists():
                # 加载已有密钥
                self._dcmh_v2_keys = self._load_dcmh_v2_keys()
                logger.info(f"已加载 DCMH V2 密钥：{self.dcmh_v2_key_path}")
            else:
                # 生成新密钥
                self._dcmh_v2_keys = self._generate_dcmh_v2_keys(bit_dim, d_prime, seed)
                self._save_dcmh_v2_keys()
                logger.info(f"已生成新 DCMH V2 密钥：{self.dcmh_v2_key_path}")

        return self._dcmh_v2_keys

    def _generate_dcmh_v2_keys(self, bit_dim: int, d_prime: Optional[int], seed: int) -> Dict[str, Any]:
        """生成 DCMH ASPE Scheme 2 密钥。"""
        from core.aspe.scheme2 import ASPEScheme2

        # ASPEScheme2.__init__ 已经在初始化时生成密钥
        aspe = ASPEScheme2(d=bit_dim, d_prime=d_prime, seed=seed)

        return {
            'M1': aspe.M1,
            'M2': aspe.M2,
            'S': aspe.S,
            'w': aspe.w,
            'M1_inv': aspe.M1_inv,
            'M2_inv': aspe.M2_inv,
            'bit_dim': bit_dim,
            'd_prime': aspe.d_prime,
            'seed': seed
        }

    def _load_dcmh_v2_keys(self) -> Dict[str, Any]:
        """加载 DCMH ASPE Scheme 2 密钥。"""
        return torch.load(self.dcmh_v2_key_path, weights_only=False)

    def _save_dcmh_v2_keys(self):
        """保存 DCMH ASPE Scheme 2 密钥。"""
        if self._dcmh_v2_keys is None:
            return

        # 确保目录存在
        self.keys_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self._dcmh_v2_keys, self.dcmh_v2_key_path)

    def _generate_dcmh_keys(self, bit_dim: int, seed: int) -> Dict[str, Any]:
        """生成 DCMH ASPE 密钥。"""
        from core.aspe.scheme1 import ASPEScheme1

        # ASPEScheme1.__init__ 已经在初始化时生成密钥
        aspe = ASPEScheme1(d=bit_dim, seed=seed)

        return {
            'M': aspe.M,
            'M_inv': aspe.M_inv,
            'bit_dim': bit_dim,
            'seed': seed
        }

    def _load_dcmh_keys(self) -> Dict[str, Any]:
        """加载 DCMH ASPE 密钥。"""
        return torch.load(self.dcmh_key_path, weights_only=False)

    def _save_dcmh_keys(self):
        """保存 DCMH ASPE 密钥。"""
        if self._dcmh_keys is None:
            return

        # 确保目录存在
        self.keys_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self._dcmh_keys, self.dcmh_key_path)

    def _generate_cir_keys(self, feature_dim: int) -> Dict[str, Any]:
        """生成 CIR ASPE 密钥。"""
        from core.aspe.cnn_wrapper import ASPEForCNN

        aspe = ASPEForCNN(feature_dim=feature_dim)
        aspe.generate_keys()

        return {
            'M1': aspe.M1,
            'M2': aspe.M2,
            'S': aspe.S,
            'M1_inv': aspe.M1_inv,
            'M2_inv': aspe.M2_inv,
            'feature_dim': feature_dim
        }

    def _load_cir_keys(self) -> Dict[str, Any]:
        """加载 CIR ASPE 密钥。"""
        return torch.load(self.cir_key_path, weights_only=False)

    def _save_cir_keys(self):
        """保存 CIR ASPE 密钥。"""
        if self._cir_keys is None:
            return

        # 确保目录存在
        self.keys_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self._cir_keys, self.cir_key_path)

    def has_dcmh_keys(self) -> bool:
        """检查 DCMH 密钥是否存在。"""
        return self.dcmh_key_path.exists()

    def has_dcmh_v2_keys(self) -> bool:
        """检查 DCMH V2 (Scheme 2) 密钥是否存在。"""
        return self.dcmh_v2_key_path.exists()

    def has_cir_keys(self) -> bool:
        """检查 CIR 密钥是否存在。"""
        return self.cir_key_path.exists()

    def get_status(self) -> Dict[str, Any]:
        """获取密钥管理器状态。"""
        return {
            "keys_dir": str(self.keys_dir),
            "dcmh_keys_exist": self.has_dcmh_keys(),
            "dcmh_v2_keys_exist": self.has_dcmh_v2_keys(),
            "cir_keys_exist": self.has_cir_keys(),
            "dcmh_keys_loaded": self._dcmh_keys is not None,
            "dcmh_v2_keys_loaded": self._dcmh_v2_keys is not None,
            "cir_keys_loaded": self._cir_keys is not None
        }

    def clear_all_keys(self):
        """
        清除所有密钥文件和内存缓存。

        警告：此操作不可逆，将导致已加密的数据无法解密！
        """
        import os

        # 清除内存缓存
        self._dcmh_keys = None
        self._dcmh_v2_keys = None
        self._cir_keys = None

        # 删除密钥文件
        if self.dcmh_key_path.exists():
            os.remove(self.dcmh_key_path)
            logger.warning(f"已删除 DCMH 密钥文件：{self.dcmh_key_path}")

        if self.dcmh_v2_key_path.exists():
            os.remove(self.dcmh_v2_key_path)
            logger.warning(f"已删除 DCMH V2 密钥文件：{self.dcmh_v2_key_path}")

        if self.cir_key_path.exists():
            os.remove(self.cir_key_path)
            logger.warning(f"已删除 CIR 密钥文件：{self.cir_key_path}")


def get_key_manager() -> KeyManager:
    """获取密钥管理器单例。"""
    return KeyManager.get_instance()