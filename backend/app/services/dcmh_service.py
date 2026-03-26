"""
DCMH 服务：深度跨模态哈希码生成

使用项目中 core/hashing/ 目录下的 DCMH 模型实现，
加载已训练好的模型，提供图像和文本哈希码生成功能。
支持多数据集（Flickr25K, NUS-WIDE）。
"""

import torch
import numpy as np
from typing import Dict, Any, Optional, List
from pathlib import Path
import sys
import glob
import os
from PIL import Image
import torchvision.transforms as transforms
import logging

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 使用项目中 core/hashing 目录下的模型定义
from core.hashing.dcmh_image import DCMHImageModule
from core.hashing.dcmh_text import DCMHTextModule

# 导入数据集配置
from backend.app.services.dataset_service import get_y_dim_for_dataset, DATASET_CONFIGS

logger = logging.getLogger(__name__)

# 模型目录配置
MODEL_DIRS = {
    'flickr25k': PROJECT_ROOT / "data" / "dcmh" / "flickr25k",
    'nuswide': PROJECT_ROOT / "data" / "dcmh" / "nuswide"
}
FALLBACK_MODEL_DIR = PROJECT_ROOT / "results"

# 预训练 VGG-F 模型路径
PRETRAIN_MODEL_PATH = PROJECT_ROOT / "data" / "imagenet-vgg-f.mat"


def preprocess_image_for_inference(image: Image.Image,
                                   target_size: tuple = (224, 224)) -> torch.Tensor:
    """
    推理时预处理图像（与训练一致）。

    步骤：
    1. 调整大小到 224x224
    2. 转换为 tensor

    注意：均值减法在模型内部完成，这里不需要手动处理。

    参数：
        image: PIL 图像对象
        target_size: 目标尺寸，默认 (224, 224)

    返回：
        预处理后的图像张量 [3, H, W]
    """
    # 转换为 RGB
    image = image.convert('RGB')

    # 调整大小
    image = image.resize(target_size, Image.BILINEAR)

    # 转换为 numpy 数组
    img_np = np.array(image, dtype=np.float32)

    # 通道顺序：HWC -> CHW
    img_np = img_np.transpose(2, 0, 1)

    # 不在这里减均值，让模型内部处理
    # 训练时模型内部会执行 x - self.mean，这里减了会导致重复减均值

    return torch.from_numpy(img_np)


def preprocess_tag_vector(tag_vector: np.ndarray) -> torch.Tensor:
    """
    预处理标签向量。

    参数：
        tag_vector: multi-hot 标签向量 [y_dim]

    返回：
        预处理后的标签张量
    """
    return torch.from_numpy(tag_vector.astype(np.float32))


class DCMHService:
    """
    DCMH 哈希码生成服务。

    提供：
    - 加载预训练模型
    - 生成图像哈希码
    - 生成文本哈希码
    - 支持多数据集
    - 正确的预处理流程
    """

    # 默认标签维度 (Flickr25K)
    DEFAULT_Y_DIM = 1386

    def __init__(self,
                 bit_dim: int = 64,
                 dataset: str = 'flickr25k',
                 img_model_path: Optional[str] = None,
                 txt_model_path: Optional[str] = None,
                 y_dim: Optional[int] = None,
                 use_gpu: bool = False):
        """
        初始化 DCMH 服务。

        参数：
            bit_dim: 哈希码位数 (16, 32, 64 等)
            dataset: 数据集名称（flickr25k, nuswide）
            img_model_path: 图像模型权重路径（None 时自动查找）
            txt_model_path: 文本模型权重路径（None 时自动查找）
            y_dim: 文本特征维度（None 时根据数据集自动设置）
            use_gpu: 是否使用 GPU
        """
        self.bit_dim = bit_dim
        self.dataset = dataset

        # 动态设置 y_dim
        if y_dim is not None:
            self.y_dim = y_dim
        else:
            self.y_dim = get_y_dim_for_dataset(dataset)

        # 自动检测 GPU
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.model_loaded = False
        self.device = torch.device("cuda" if self.use_gpu else "cpu")

        # 初始化图像模型
        self.img_model = DCMHImageModule(bit=bit_dim, pretrain_model=None)

        # 初始化文本模型
        self.txt_model = DCMHTextModule(y_dim=self.y_dim, bit=bit_dim)

        # 自动查找最新训练的模型
        if img_model_path is None:
            img_model_path = self._find_latest_model('img_model')
        if txt_model_path is None:
            txt_model_path = self._find_latest_model('txt_model')

        # 加载预训练权重
        if img_model_path and os.path.exists(img_model_path):
            self._load_img_model(img_model_path)

        if txt_model_path and os.path.exists(txt_model_path):
            self._load_txt_model(txt_model_path)

        # 设置设备
        if self.use_gpu:
            self.img_model = self.img_model.cuda()
            self.txt_model = self.txt_model.cuda()

        # 设置为评估模式
        self.img_model.eval()
        self.txt_model.eval()

        logger.info(f"[DCMHService] 初始化完成：dataset={dataset}, bit_dim={bit_dim}, y_dim={self.y_dim}")

    def _find_latest_model(self, model_type: str) -> Optional[str]:
        """
        查找最新训练的模型文件。

        查找顺序：
        1. data/dcmh/{dataset}/ (部署目录)
        2. results/{dataset}/*/ (训练目录)

        参数：
            model_type: 'img_model' 或 'txt_model'

        返回：
            模型文件路径，如果未找到返回 None
        """
        # 获取当前数据集的模型目录
        model_dir = MODEL_DIRS.get(self.dataset)
        fallback_dir = FALLBACK_MODEL_DIR / self.dataset

        search_dirs = []
        if model_dir and model_dir.exists():
            search_dirs.append(model_dir)
        if fallback_dir.exists():
            search_dirs.append(fallback_dir)

        for model_dir in search_dirs:
            # 部署目录直接查找模型文件
            if "dcmh" in str(model_dir):
                model_file = model_dir / f"{model_type}.pth"
                if model_file.exists():
                    logger.info(f"[DCMHService] 使用部署模型 {model_type}：{model_file}")
                    return str(model_file)

            # 训练目录查找子目录中的模型
            pattern = str(model_dir / "*" / f"{model_type}.pth")
            import glob as glob_module
            model_files = glob_module.glob(pattern)

            if model_files:
                # 按修改时间排序，选择最新的
                model_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
                latest_model = model_files[0]
                logger.info(f"[DCMHService] 自动选择最新 {model_type}：{latest_model}")
                return latest_model

        logger.warning(f"[DCMHService] 未找到 {model_type} 模型文件（数据集：{self.dataset}）")
        return None

    def _load_img_model(self, path: str):
        """加载图像模型权重。"""
        try:
            # 使用 map_location 确保在 CPU 上也能加载 GPU 训练的模型
            self.img_model.load(path, use_gpu=self.use_gpu)
            logger.info(f"[DCMHService] 已加载图像模型：{path}")
            self.model_loaded = True
        except Exception as e:
            logger.error(f"[DCMHService] 加载图像模型失败：{e}")

    def _load_txt_model(self, path: str):
        """加载文本模型权重。"""
        try:
            self.txt_model.load(path, use_gpu=self.use_gpu)
            logger.info(f"[DCMHService] 已加载文本模型：{path}")
            self.model_loaded = True
        except Exception as e:
            logger.error(f"[DCMHService] 加载文本模型失败：{e}")

    def generate_image_code(self, images: torch.Tensor) -> torch.Tensor:
        """
        生成图像哈希码。

        参数：
            images: 图像张量 [N, 3, H, W]，已预处理和归一化

        返回：
            哈希码张量 [N, bit_dim]，值为 {-1, +1}
        """
        batch_size = 64
        num_data = images.shape[0]
        index = np.linspace(0, num_data - 1, num_data).astype(int)
        B = torch.zeros(num_data, self.bit_dim, dtype=torch.float)

        if self.use_gpu:
            B = B.cuda()
            images = images.cuda()

        with torch.no_grad():
            for i in range(num_data // batch_size + 1):
                ind = index[i * batch_size: min((i + 1) * batch_size, num_data)]
                image_batch = images[ind].type(torch.float)

                cur_f = self.img_model(image_batch)
                B[ind, :] = cur_f.data

        B = torch.sign(B)
        return B

    def generate_text_code(self, texts: torch.Tensor) -> torch.Tensor:
        """
        生成文本哈希码。

        参数：
            texts: 文本张量 [N, seq_len] 或 [N]（索引形式）

        返回：
            哈希码张量 [N, bit_dim]，值为 {-1, +1}
        """
        batch_size = 64
        num_data = texts.shape[0]
        index = np.linspace(0, num_data - 1, num_data).astype(int)
        B = torch.zeros(num_data, self.bit_dim, dtype=torch.float)

        if self.use_gpu:
            B = B.cuda()
            texts = texts.cuda()

        with torch.no_grad():
            for i in range(num_data // batch_size + 1):
                ind = index[i * batch_size: min((i + 1) * batch_size, num_data)]
                text_batch = texts[ind]

                # 文本数据需要额外的维度
                if text_batch.dim() == 2:
                    text_batch = text_batch.unsqueeze(1).unsqueeze(-1)
                elif text_batch.dim() == 1:
                    text_batch = text_batch.unsqueeze(1).unsqueeze(-1)

                text_batch = text_batch.type(torch.float)
                cur_g = self.txt_model(text_batch)
                B[ind, :] = cur_g.data

        B = torch.sign(B)
        return B

    def generate_image_code_single(self, image: torch.Tensor) -> torch.Tensor:
        """生成单个图像的哈希码。"""
        if image.dim() == 3:
            image = image.unsqueeze(0)

        with torch.no_grad():
            if self.use_gpu:
                image = image.cuda()
            code = self.img_model(image)
            code = torch.sign(code)

        return code

    def generate_text_code_single(self, text: torch.Tensor) -> torch.Tensor:
        """生成单个文本的哈希码。"""
        if text.dim() == 1:
            text = text.unsqueeze(0)

        with torch.no_grad():
            if self.use_gpu:
                text = text.cuda()

            # 文本数据需要额外的维度
            if text.dim() == 2:
                text = text.unsqueeze(1).unsqueeze(-1)

            code = self.txt_model(text)
            code = torch.sign(code)

        return code

    def generate_text_code_from_tags(self,
                                        tag_indices: List[int],
                                        y_dim: int) -> torch.Tensor:
        """
        从标签索引生成文本哈希码。

        参数：
            tag_indices: 标签索引列表
            y_dim: 标签维度

        返回：
            哈希码张量 [1, bit_dim]
        """
        # 构建 multi-hot 向量
        tag_vector = np.zeros(y_dim, dtype=np.float32)
        for idx in tag_indices:
            if 0 <= idx < y_dim:
                tag_vector[idx] = 1.0

        # 生成哈希码
        text_tensor = torch.from_numpy(tag_vector).unsqueeze(0)
        return self.generate_text_code_single(text_tensor)

    def generate_database_codes(self, images: torch.Tensor, batch_size: int = 64) -> torch.Tensor:
        """
        批量生成数据库哈希码。

        参数：
            images: 图像张量 [N, 3, H, W]
            batch_size: 批次大小

        返回：
            哈希码张量 [N, bit_dim]
        """
        return self.generate_image_code(images)

    def is_loaded(self) -> bool:
        """检查模型是否已加载。"""
        return self.model_loaded

    def get_status(self) -> Dict[str, Any]:
        """获取服务状态。"""
        return {
            "bit_dim": self.bit_dim,
            "y_dim": self.y_dim,
            "dataset": self.dataset,
            "use_gpu": self.use_gpu,
            "model_loaded": self.model_loaded,
            "device": "cuda" if self.use_gpu else "cpu",
            "gpu_name": torch.cuda.get_device_name(0) if self.use_gpu else None
        }


# 全局服务实例缓存（按数据集名称）
_dcmh_services: Dict[str, DCMHService] = {}


def get_dcmh_service(dataset: str = 'flickr25k', **kwargs) -> DCMHService:
    """
    获取或创建 DCMH 服务实例。

    参数：
        dataset: 数据集名称（flickr25k, nuswide）

    返回：
        DCMHService 实例
    """
    if dataset not in _dcmh_services:
        _dcmh_services[dataset] = DCMHService(dataset=dataset, **kwargs)
    return _dcmh_services[dataset]


def get_all_dcmh_services() -> Dict[str, DCMHService]:
    """获取所有 DCMH 服务实例。"""
    return _dcmh_services.copy()