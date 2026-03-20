"""
COCO 服务：MS-COCO 数据集管理

封装 MS-COCO 数据集加载和管理功能。
"""

import os
import json
import torch
import numpy as np
from PIL import Image
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import torchvision.transforms as transforms

# 添加项目根目录到路径
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from config.coco_config import COCO_CONFIG


class COCOService:
    """
    MS-COCO 数据集服务。

    提供：
    - 加载数据集
    - 获取图像和文本
    - 数据预处理
    - 检索数据库构建
    """

    def __init__(self,
                 data_root: Optional[str] = None,
                 images_dir: Optional[str] = None,
                 annotations_file: Optional[str] = None,
                 image_size: int = 224):
        """
        初始化 COCO 服务。

        参数：
            data_root: COCO 数据根目录
            images_dir: 图像目录
            annotations_file: 注释文件路径
            image_size: 图像预处理大小
        """
        self.data_root = data_root or COCO_CONFIG.get('data_root', './data/coco')
        self.images_dir = images_dir or COCO_CONFIG.get('images_dir', './data/coco/images')
        self.annotations_file = annotations_file or COCO_CONFIG.get(
            'annotations_file', './data/coco/annotations/captions_train2014.json'
        )
        self.image_size = image_size

        # 数据缓存
        self.images_data: Optional[List[Dict]] = None
        self.annotations_data: Optional[List[Dict]] = None
        self.id_to_filename: Dict[int, str] = {}
        self.image_to_annotations: Dict[int, List[Dict]] = {}

        # 图像变换
        self.transform = self._get_transform()

        # 文本词汇表
        self.vocab: Optional[Dict[str, int]] = None

    def _get_transform(self) -> transforms.Compose:
        """获取图像预处理变换。"""
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def load_data(self) -> bool:
        """
        加载 COCO 数据集。

        返回：
            是否加载成功
        """
        try:
            # 检查文件是否存在
            if not os.path.exists(self.annotations_file):
                print(f"注释文件不存在：{self.annotations_file}")
                # 创建模拟数据用于演示
                self._create_mock_data()
                return True

            print(f"正在加载 COCO 数据集：{self.annotations_file}")
            with open(self.annotations_file, 'r', encoding='utf-8') as f:
                coco_data = json.load(f)

            self.images_data = coco_data.get('images', [])
            self.annotations_data = coco_data.get('annotations', [])

            # 建立映射
            self.id_to_filename = {img['id']: img['file_name'] for img in self.images_data}

            for ann in self.annotations_data:
                image_id = ann['image_id']
                if image_id not in self.image_to_annotations:
                    self.image_to_annotations[image_id] = []
                self.image_to_annotations[image_id].append(ann)

            print(f"已加载 {len(self.images_data)} 张图像，{len(self.annotations_data)} 条注释")
            return True

        except Exception as e:
            print(f"加载 COCO 数据集失败：{e}")
            self._create_mock_data()
            return True

    def _create_mock_data(self):
        """创建模拟数据用于演示（当真实数据不可用时）。"""
        print("使用模拟数据进行演示...")

        # 创建模拟图像数据
        self.images_data = []
        for i in range(100):
            self.images_data.append({
                'id': i,
                'file_name': f'image_{i:04d}.jpg',
                'width': 640,
                'height': 480
            })

        # 创建模拟注释
        self.annotations_data = []
        mock_captions = [
            "A cat sitting on a chair",
            "A dog playing in the park",
            "A beautiful sunset over the ocean",
            "People walking on the street",
            "A bird flying in the sky",
            "A car parked on the road",
            "A child eating ice cream",
            "A flower blooming in the garden",
            "A book on the table",
            "A computer on the desk"
        ]

        for i in range(100):
            for j in range(5):  # 每张图像 5 个标题
                self.annotations_data.append({
                    'id': i * 5 + j,
                    'image_id': i,
                    'caption': mock_captions[j % len(mock_captions)]
                })

        self.id_to_filename = {img['id']: img['file_name'] for img in self.images_data}

        for ann in self.annotations_data:
            image_id = ann['image_id']
            if image_id not in self.image_to_annotations:
                self.image_to_annotations[image_id] = []
            self.image_to_annotations[image_id].append(ann)

        print(f"已创建 {len(self.images_data)} 张模拟图像")

    def get_image(self, image_id: int) -> Tuple[Optional[torch.Tensor], Dict]:
        """
        获取单个图像。

        参数：
            image_id: 图像 ID

        返回：
            (图像张量，元数据字典)
        """
        if image_id not in self.id_to_filename:
            return None, {'error': 'Image not found'}

        filename = self.id_to_filename[image_id]
        image_path = os.path.join(self.images_dir, filename)

        # 检查是否是模拟数据
        if not os.path.exists(image_path):
            # 生成模拟图像
            image = self._generate_mock_image(image_id)
        else:
            try:
                image = Image.open(image_path).convert('RGB')
                image = self.transform(image)
            except Exception as e:
                print(f"加载图像失败：{e}")
                image = self._generate_mock_image(image_id)

        metadata = {
            'image_id': image_id,
            'filename': filename,
            'num_captions': len(self.image_to_annotations.get(image_id, []))
        }

        return image, metadata

    def _generate_mock_image(self, image_id: int) -> torch.Tensor:
        """生成模拟图像张量。"""
        # 使用随机但可重复的颜色
        np.random.seed(image_id)
        image_array = np.random.rand(3, self.image_size, self.image_size).astype(np.float32)
        image_array = (image_array - image_array.min()) / (image_array.max() - image_array.min())
        return torch.from_numpy(image_array)

    def get_captions(self, image_id: int) -> List[str]:
        """获取图像的所有标题。"""
        annotations = self.image_to_annotations.get(image_id, [])
        return [ann['caption'] for ann in annotations]

    def get_all_images(self, limit: int = 100) -> List[Dict]:
        """获取所有图像的元数据列表。"""
        result = []
        for img in self.images_data[:limit]:
            image_id = img['id']
            result.append({
                'id': image_id,
                'filename': self.id_to_filename.get(image_id, ''),
                'num_captions': len(self.image_to_annotations.get(image_id, 0)),
                'captions': self.get_captions(image_id)
            })
        return result

    def get_dataset_statistics(self) -> Dict[str, Any]:
        """获取数据集统计信息。"""
        num_images = len(self.images_data)
        num_annotations = len(self.annotations_data)
        num_unique_captions = len(set(ann['caption'] for ann in self.annotations_data))

        # 计算平均 caption 长度
        caption_lengths = [len(ann['caption'].split()) for ann in self.annotations_data]
        avg_caption_length = np.mean(caption_lengths) if caption_lengths else 0

        return {
            'num_images': num_images,
            'num_annotations': num_annotations,
            'num_unique_captions': num_unique_captions,
            'avg_caption_length': avg_caption_length,
            'captions_per_image': num_annotations / num_images if num_images > 0 else 0
        }

    def get_retrieval_data(self,
                          limit: int = 100) -> Dict[str, Any]:
        """
        获取用于检索的数据。

        返回：
            包含图像、文本、标签的字典
        """
        if not self.images_data:
            self.load_data()

        # 获取图像 ID 列表
        image_ids = [img['id'] for img in self.images_data[:limit]]

        # 为每个图像分配标签（模拟）
        # 实际应用中应该使用真实的多标签
        labels = []
        for i, image_id in enumerate(image_ids):
            # 模拟 2000 维标签，这里简化为 10 维
            label = np.zeros(10, dtype=np.float32)
            label[i % 10] = 1.0  # 简单分配
            labels.append(label)

        labels = np.array(labels)

        return {
            'image_ids': image_ids,
            'num_images': len(image_ids),
            'labels': labels.tolist(),
            'label_dim': labels.shape[1]
        }

    def text_to_vector(self, text: str) -> torch.Tensor:
        """
        将文本转换为向量（简化版本）。

        实际应用中应该使用 BERT 或其他文本编码器。
        """
        # 简单的词袋模型用于演示
        words = text.lower().split()
        vector = np.zeros(2000, dtype=np.float32)

        if self.vocab is None:
            # 创建简单词汇表
            self.vocab = {word: i for i, word in enumerate(set(
                word for ann in self.annotations_data for word in ann['caption'].lower().split()
            )[:1999])}

        for word in words:
            if word in self.vocab:
                vector[self.vocab[word]] = 1.0

        return torch.from_numpy(vector)


# 全局服务实例
_coco_service: Optional[COCOService] = None


def get_coco_service(**kwargs) -> COCOService:
    """获取或创建 COCO 服务单例。"""
    global _coco_service
    if _coco_service is None:
        _coco_service = COCOService(**kwargs)
        _coco_service.load_data()
    return _coco_service
