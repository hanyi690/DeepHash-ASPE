"""
DCMH 服务：深度跨模态哈希码生成

封装 DCMH 参考实现，提供图像和文本哈希码生成功能。
"""

import torch
import numpy as np
from typing import Dict, Any, Optional
from pathlib import Path
import sys

# 添加项目根目录和 reference/DCMH 到路径
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "reference" / "DCMH"))

# 现在可以导入 DCMH 模块
from models.img_module import ImgModule
from models.txt_module import TxtModule


class DCMHService:
    """
    DCMH 哈希码生成服务。

    提供：
    - 加载预训练模型
    - 生成图像哈希码
    - 生成文本哈希码
    """

    def __init__(self,
                 bit_dim: int = 64,
                 img_model_path: Optional[str] = None,
                 txt_model_path: Optional[str] = None,
                 use_gpu: bool = False):
        """
        初始化 DCMH 服务。

        参数：
            bit_dim: 哈希码位数 (16, 32, 64 等)
            img_model_path: 图像模型权重路径
            txt_model_path: 文本模型权重路径
            use_gpu: 是否使用 GPU
        """
        self.bit_dim = bit_dim
        self.use_gpu = use_gpu and torch.cuda.is_available()

        # 初始化模型
        self.img_model = ImgModule(bit_dim)
        self.txt_model = TxtModule(y_dim=2000, bit=bit_dim)  # y_dim 需根据实际标签维度调整

        # 加载预训练权重
        if img_model_path:
            self._load_img_model(img_model_path)

        if txt_model_path:
            self._load_txt_model(txt_model_path)

        # 设置设备
        if self.use_gpu:
            self.img_model = self.img_model.cuda()
            self.txt_model = self.txt_model.cuda()

        # 设置为评估模式
        self.img_model.eval()
        self.txt_model.eval()

    def _load_img_model(self, path: str):
        """加载图像模型权重。"""
        try:
            self.img_model.load(path)
            print(f"已加载图像模型：{path}")
        except Exception as e:
            print(f"加载图像模型失败：{e}")

    def _load_txt_model(self, path: str):
        """加载文本模型权重。"""
        try:
            self.txt_model.load(path)
            print(f"已加载文本模型：{path}")
        except Exception as e:
            print(f"加载文本模型失败：{e}")

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


# 全局服务实例（单例模式）
_dcmh_service: Optional[DCMHService] = None


def get_dcmh_service(**kwargs) -> DCMHService:
    """获取或创建 DCMH 服务单例。"""
    global _dcmh_service
    if _dcmh_service is None:
        _dcmh_service = DCMHService(**kwargs)
    return _dcmh_service
