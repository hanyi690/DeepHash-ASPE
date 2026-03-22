"""CNN Image Retrieval Service for Backend API."""

from typing import List, Dict, Optional, Any
from dataclasses import dataclass, asdict
import os
from pathlib import Path
import sys
import torch

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.retrieval.cir_service import CIRService as CoreCIRService, RetrievalResult
from core.retrieval.feature_extractor import FeatureExtractor


@dataclass
class ImageSearchResult:
    """Image search result for API response."""
    image_id: str
    image_path: str
    score: float
    rank: int


class CIRService:
    """
    Backend service for CNN Image Retrieval.

    This service wraps the core CIR functionality for use in the FastAPI backend.
    """

    _instance: Optional['CIRService'] = None
    _initialized: bool = False

    def __init__(self):
        """Initialize the CIR service."""
        self.core_service: Optional[CoreCIRService] = None
        self.feature_extractor: Optional[FeatureExtractor] = None
        self.index_dir: Optional[Path] = None

    @classmethod
    def get_instance(cls) -> 'CIRService':
        """Get singleton instance of the service."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def initialize(
        self,
        architecture: str = 'resnet50',
        pooling: str = 'gem',
        whitening: bool = True,
        index_dir: Optional[str] = None
    ):
        """
        Initialize the CIR service.

        Args:
            architecture: Backbone architecture (resnet50, vgg16, etc.)
            pooling: Pooling method (gem, mac, spoc, rmac)
            whitening: Whether to use whitening
            index_dir: Directory to load/save index
        """
        if self._initialized:
            return

        print(f"[CIRService] Initializing with {architecture}-{pooling}...")

        # Create core service
        self.core_service = CoreCIRService(
            architecture=architecture,
            pooling=pooling,
            whitening=whitening
        )

        # Create standalone feature extractor
        self.feature_extractor = FeatureExtractor(
            architecture=architecture,
            pooling=pooling,
            whitening=whitening
        )

        # Load index if provided
        if index_dir and os.path.exists(index_dir):
            self.core_service.load_index(index_dir)
            print(f"[CIRService] Index loaded from {index_dir}")
        else:
            self.index_dir = Path(index_dir) if index_dir else None

        self._initialized = True
        print(f"[CIRService] Initialized successfully")

    def build_index(
        self,
        image_paths: List[str],
        image_ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Build feature index from images.

        Args:
            image_paths: List of image paths to index
            image_ids: Optional list of image IDs

        Returns:
            Status dictionary with index info
        """
        if not self._initialized:
            self.initialize()

        if not image_paths:
            return {"status": "error", "message": "No image paths provided"}

        try:
            self.core_service.build_index(image_paths, image_ids)
            return {
                "status": "success",
                "num_images": len(image_paths),
                "feature_dim": self.core_service.get_feature_dim()
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def search(
        self,
        query_image_path: str,
        top_k: int = 10
    ) -> Dict[str, Any]:
        """
        Search for similar images.

        Args:
            query_image_path: Path to query image
            top_k: Number of results to return

        Returns:
            Dictionary with search results
        """
        if not self._initialized:
            return {"status": "error", "message": "Service not initialized"}

        if not self.core_service.is_indexed:
            return {"status": "error", "message": "Index not built"}

        try:
            results = self.core_service.search(query_image_path, top_k=top_k)
            return {
                "status": "success",
                "results": [asdict(r) for r in results]
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def extract_feature(self, image_path: str) -> Dict[str, Any]:
        """
        Extract feature vector from an image.

        Args:
            image_path: Path to the image

        Returns:
            Dictionary with feature vector info
        """
        if not self._initialized:
            return {"status": "error", "message": "Service not initialized"}

        try:
            feature = self.feature_extractor.extract(image_path)
            return {
                "status": "success",
                "feature_dim": feature.shape[0],
                "feature_norm": float(torch.norm(feature).item())
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def save_index(self, save_dir: str) -> Dict[str, Any]:
        """Save the current index to disk."""
        if not self._initialized or not self.core_service.is_indexed:
            return {"status": "error", "message": "No index to save"}

        try:
            self.core_service.save_index(save_dir)
            return {"status": "success", "save_dir": save_dir}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def load_index(self, index_dir: str) -> Dict[str, Any]:
        """Load index from disk."""
        if not self._initialized:
            self.initialize()

        try:
            self.core_service.load_index(index_dir)
            return {
                "status": "success",
                "num_images": self.core_service.get_index_size()
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def get_status(self) -> Dict[str, Any]:
        """Get service status."""
        return {
            "initialized": self._initialized,
            "indexed": self.core_service.is_indexed if self.core_service else False,
            "index_size": self.core_service.get_index_size() if self.core_service else 0,
            "feature_dim": self.core_service.get_feature_dim() if self.core_service else 0
        }


# Singleton accessor
def get_cir_service() -> CIRService:
    """Get the CIR service singleton."""
    return CIRService.get_instance()
