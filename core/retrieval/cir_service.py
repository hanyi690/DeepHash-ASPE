"""
CNN Image Retrieval Service

This module provides a service layer for CNN-based image retrieval,
including feature indexing and similarity search.
"""

import os
import torch
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass
import pickle

from .feature_extractor import FeatureExtractor


@dataclass
class RetrievalResult:
    """Represents a single retrieval result."""
    image_id: str
    image_path: str
    score: float
    rank: int


class CIRService:
    """
    CNN Image Retrieval Service.

    Provides end-to-end image retrieval functionality including:
    - Building feature indices from image collections
    - Querying with images to find similar images
    - Managing feature databases
    """

    def __init__(
        self,
        architecture: str = 'resnet50',
        pooling: str = 'gem',
        whitening: bool = True,
        device: Optional[str] = None
    ):
        """
        Initialize the CIR service.

        Args:
            architecture: Backbone architecture for feature extraction
            pooling: Pooling method (mac, spoc, gem, rmac)
            whitening: Whether to use whitening
            device: Device to run computations on
        """
        self.feature_extractor = FeatureExtractor(
            architecture=architecture,
            pooling=pooling,
            whitening=whitening,
            device=device
        )

        # Index storage
        self.features: Optional[torch.Tensor] = None
        self.image_ids: List[str] = []
        self.image_paths: List[str] = []
        self.is_indexed = False

    def build_index(
        self,
        image_paths: List[str],
        image_ids: Optional[List[str]] = None,
        batch_size: int = 32
    ):
        """
        Build the feature index from a collection of images.

        Args:
            image_paths: List of paths to images in the database
            image_ids: Optional list of unique IDs for each image.
                      If None, uses indices as IDs.
            batch_size: Batch size for feature extraction
        """
        print(f"[CIRService] Building index for {len(image_paths)} images...")

        # Set image IDs
        if image_ids is None:
            self.image_ids = [f"img_{i}" for i in range(len(image_paths))]
        else:
            self.image_ids = image_ids

        self.image_paths = image_paths

        # Extract features
        self.features = self.feature_extractor.extract_batch(
            image_paths, batch_size=batch_size
        )

        self.is_indexed = True
        print(f"[CIRService] Index built successfully. Feature shape: {self.features.shape}")

    def search(
        self,
        query_image_path: str,
        top_k: int = 10,
        exclude_query: bool = False
    ) -> List[RetrievalResult]:
        """
        Search for similar images given a query image.

        Args:
            query_image_path: Path to the query image
            top_k: Number of results to return
            exclude_query: Whether to exclude the query image from results

        Returns:
            List of RetrievalResult objects sorted by similarity
        """
        if not self.is_indexed:
            raise RuntimeError("Index not built. Call build_index() first.")

        # Extract query feature
        query_feature = self.feature_extractor.extract(query_image_path)

        # Compute similarities (cosine similarity via dot product since features are L2-normalized)
        similarities = torch.matmul(self.features.t(), query_feature)

        # Get top-k indices
        scores, indices = torch.topk(similarities, k=top_k + int(exclude_query))

        results = []
        for i, (score, idx) in enumerate(zip(scores, indices)):
            img_idx = idx.item()

            # Skip if this is the query image
            if exclude_query and self.image_paths[img_idx] == query_image_path:
                continue

            results.append(RetrievalResult(
                image_id=self.image_ids[img_idx],
                image_path=self.image_paths[img_idx],
                score=score.item(),
                rank=len(results)
            ))

            if len(results) >= top_k:
                break

        return results

    def search_by_feature(
        self,
        query_feature: torch.Tensor,
        top_k: int = 10
    ) -> List[RetrievalResult]:
        """
        Search for similar images given a pre-extracted feature.

        Args:
            query_feature: Pre-extracted feature vector
            top_k: Number of results to return

        Returns:
            List of RetrievalResult objects sorted by similarity
        """
        if not self.is_indexed:
            raise RuntimeError("Index not built. Call build_index() first.")

        # Ensure feature is 1D
        if query_feature.dim() > 1:
            query_feature = query_feature.squeeze()

        # Compute similarities
        similarities = torch.matmul(self.features.t(), query_feature)

        # Get top-k indices
        scores, indices = torch.topk(similarities, k=top_k)

        results = []
        for score, idx in zip(scores, indices):
            img_idx = idx.item()
            results.append(RetrievalResult(
                image_id=self.image_ids[img_idx],
                image_path=self.image_paths[img_idx],
                score=score.item(),
                rank=len(results)
            ))

        return results

    def save_index(self, save_dir: str):
        """
        Save the index to disk.

        Args:
            save_dir: Directory to save the index files
        """
        if not self.is_indexed:
            raise RuntimeError("No index to save. Call build_index() first.")

        os.makedirs(save_dir, exist_ok=True)

        # Save features
        torch.save(self.features, os.path.join(save_dir, 'features.pt'))

        # Save metadata
        metadata = {
            'image_ids': self.image_ids,
            'image_paths': self.image_paths,
            'model_meta': self.feature_extractor.get_meta()
        }
        with open(os.path.join(save_dir, 'metadata.pkl'), 'wb') as f:
            pickle.dump(metadata, f)

        print(f"[CIRService] Index saved to {save_dir}")

    def load_index(self, index_dir: str, device: Optional[str] = None):
        """
        Load an index from disk.

        Args:
            index_dir: Directory containing the index files
            device: Device to load tensors to. If None, uses the current device setting.
        """
        features_path = os.path.join(index_dir, 'features.pt')
        metadata_path = os.path.join(index_dir, 'metadata.pkl')

        if not os.path.exists(features_path) or not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Index files not found in {index_dir}")

        if device is None:
            device = self.feature_extractor.device

        # Load features
        self.features = torch.load(features_path, map_location=device)

        # Load metadata
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)

        self.image_ids = metadata['image_ids']
        self.image_paths = metadata['image_paths']
        self.is_indexed = True

        print(f"[CIRService] Index loaded from {index_dir}. Feature shape: {self.features.shape}")

    def get_feature(self, image_path: str) -> torch.Tensor:
        """
        Extract feature for a single image without indexing.

        Args:
            image_path: Path to the image

        Returns:
            Feature vector
        """
        return self.feature_extractor.extract(image_path)

    def get_index_size(self) -> int:
        """Get the number of indexed images."""
        return len(self.image_paths) if self.is_indexed else 0

    def get_feature_dim(self) -> int:
        """Get the feature dimension."""
        return self.feature_extractor.get_output_dim()
