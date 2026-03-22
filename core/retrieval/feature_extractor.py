"""
CNN Image Retrieval Feature Extractor

This module provides a unified interface for extracting image features
using CNN-based image retrieval models (cirtorch).
"""

import os
import torch
import torchvision.transforms as transforms
from PIL import Image
from typing import List, Union, Optional
import numpy as np

from ..cirtorch.networks.imageretrievalnet import init_network, extract_vectors


class FeatureExtractor:
    """
    CNN-based feature extractor for image retrieval.

    Supports various backbone architectures (ResNet, VGG, DenseNet, etc.)
    and pooling methods (MAC, SPoC, GeM, RMAC).
    """

    # Default preprocessing transforms matching cirtorch requirements
    DEFAULT_MEAN = [0.485, 0.456, 0.406]
    DEFAULT_STD = [0.229, 0.224, 0.225]
    DEFAULT_IMAGE_SIZE = 1024

    def __init__(
        self,
        architecture: str = 'resnet50',
        pooling: str = 'gem',
        whitening: bool = True,
        local_whitening: bool = False,
        regional: bool = False,
        pretrained: bool = True,
        device: Optional[str] = None
    ):
        """
        Initialize the feature extractor.

        Args:
            architecture: Backbone architecture name.
                         Options: resnet18/34/50/101/152, vgg11/13/16/19,
                                  densenet121/169/201/161, alexnet,
                                  squeezenet1_0/1_1
            pooling: Pooling method. Options: mac, spoc, gem, gemmp, rmac
            whitening: Whether to use learned whitening layer
            local_whitening: Whether to use local whitening before pooling
            regional: Whether to use regional pooling
            pretrained: Whether to load pretrained weights
            device: Device to run model on. If None, auto-select GPU if available
        """
        self.architecture = architecture
        self.pooling = pooling
        self.whitening = whitening
        self.local_whitening = local_whitening
        self.regional = regional
        self.pretrained = pretrained

        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        # Initialize model
        self.model = None
        self.transform = None
        self._initialize_model()
        self._initialize_transform()

    def _initialize_model(self):
        """Initialize the CNN image retrieval model."""
        params = {
            'architecture': self.architecture,
            'pooling': self.pooling,
            'whitening': self.whitening,
            'local_whitening': self.local_whitening,
            'regional': self.regional,
            'pretrained': self.pretrained
        }

        print(f"[FeatureExtractor] Initializing network: {self.architecture}-{self.pooling}")
        self.model = init_network(params)
        self.model.to(self.device)
        self.model.eval()
        print(f"[FeatureExtractor] Model initialized on {self.device}")

    def _initialize_transform(self):
        """Initialize image preprocessing transforms."""
        self.transform = transforms.Compose([
            transforms.Resize(self.DEFAULT_IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.DEFAULT_MEAN, std=self.DEFAULT_STD)
        ])

    def extract(self, image_path: str) -> torch.Tensor:
        """
        Extract feature vector from a single image.

        Args:
            image_path: Path to the input image

        Returns:
            Feature vector as a 1D tensor (L2-normalized)
        """
        # Load and preprocess image
        img = Image.open(image_path).convert('RGB')
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)

        # Extract feature
        with torch.no_grad():
            feature = self.model(img_tensor)

        return feature.squeeze()

    def extract_batch(self, image_paths: List[str], batch_size: int = 32) -> torch.Tensor:
        """
        Extract feature vectors from multiple images.

        Args:
            image_paths: List of image paths
            batch_size: Batch size for processing

        Returns:
            Feature matrix of shape (output_dim, num_images)
        """
        num_images = len(image_paths)
        output_dim = self.model.meta['outputdim']
        features = torch.zeros(output_dim, num_images)

        # Process in batches
        for i in range(0, num_images, batch_size):
            batch_paths = image_paths[i:i+batch_size]
            batch_tensors = []

            for path in batch_paths:
                img = Image.open(path).convert('RGB')
                img_tensor = self.transform(img)
                batch_tensors.append(img_tensor)

            batch = torch.stack(batch_tensors).to(self.device)

            with torch.no_grad():
                batch_features = self.model(batch)

            batch_size_actual = batch_features.size(1)
            features[:, i:i+batch_size_actual] = batch_features

            if (i // batch_size + 1) % 10 == 0:
                print(f"[FeatureExtractor] Processed {min(i+batch_size, num_images)}/{num_images} images")

        print(f"[FeatureExtractor] Extracted features for {num_images} images")
        return features

    def extract_from_pil(self, image: Image.Image) -> torch.Tensor:
        """
        Extract feature from a PIL image object.

        Args:
            image: PIL Image object

        Returns:
            Feature vector as a 1D tensor
        """
        if image.mode != 'RGB':
            image = image.convert('RGB')

        img_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            feature = self.model(img_tensor)

        return feature.squeeze()

    def get_output_dim(self) -> int:
        """Get the output feature dimension."""
        return self.model.meta['outputdim']

    def get_meta(self) -> dict:
        """Get model metadata."""
        return self.model.meta.copy()
