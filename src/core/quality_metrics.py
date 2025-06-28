"""
Quality metrics processor for Meme-Cleanup.

Handles calculation of BRISQUE and NIQE quality metrics for image quality assessment.
"""

import logging
import numpy as np
import cv2
from pathlib import Path
from typing import Optional, Tuple
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

from utils.image_utils import load_image


logger = logging.getLogger(__name__)


class QualityMetricsProcessor:
    """Handles image quality metric calculations."""
    
    def __init__(self):
        """Initialize quality metrics processor."""
        logger.info("Initializing quality metrics processor")
    
    def calculate_brisque_score(self, image_path: Path) -> Optional[float]:
        """
        Calculate BRISQUE (Blind/Referenceless Image Spatial Quality Evaluator) score.
        
        Note: This is a simplified implementation. For production use, consider using
        a pre-trained BRISQUE model or the full implementation.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            BRISQUE score (lower is better) or None if failed
        """
        try:
            # Load image
            image = load_image(image_path)
            if image is None:
                return None
            
            # Convert to grayscale for BRISQUE calculation
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # Apply Gaussian filter
            blurred = cv2.GaussianBlur(gray, (7, 7), 1.166)
            
            # Calculate MSCN (Mean Subtracted Contrast Normalized) coefficients
            mscn = (gray.astype(np.float64) - blurred.astype(np.float64)) / (blurred.astype(np.float64) + 1e-8)
            
            # Calculate local variance
            mu = cv2.GaussianBlur(mscn, (7, 7), 1.166)
            sigma = np.sqrt(cv2.GaussianBlur(mscn**2, (7, 7), 1.166) - mu**2)
            
            # Calculate normalized MSCN
            normalized_mscn = mscn / (sigma + 1e-8)
            
            # Calculate BRISQUE features (simplified)
            # In a full implementation, these would be more sophisticated
            alpha = 0.1
            beta = 0.1
            
            # Calculate statistics
            mean_mscn = np.mean(normalized_mscn)
            var_mscn = np.var(normalized_mscn)
            
            # Simplified BRISQUE score (lower is better quality)
            brisque_score = alpha * abs(mean_mscn) + beta * var_mscn
            
            return float(brisque_score)
            
        except Exception as e:
            logger.error(f"Failed to calculate BRISQUE score for {image_path}: {e}")
            return None
    
    def calculate_niqe_score(self, image_path: Path) -> Optional[float]:
        """
        Calculate NIQE (Natural Image Quality Evaluator) score.
        
        Note: This is a simplified implementation. For production use, consider using
        a pre-trained NIQE model or the full implementation.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            NIQE score (lower is better) or None if failed
        """
        try:
            # Load image
            image = load_image(image_path)
            if image is None:
                return None
            
            # Convert to YUV color space
            yuv = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            y_channel = yuv[:, :, 0]
            
            # Apply DCT transform
            dct = cv2.dct(y_channel.astype(np.float32))
            
            # Calculate NIQE features (simplified)
            # In a full implementation, these would be more sophisticated
            alpha = 0.05
            beta = 0.05
            
            # Calculate statistics
            mean_dct = np.mean(np.abs(dct))
            var_dct = np.var(np.abs(dct))
            
            # Simplified NIQE score (lower is better quality)
            niqe_score = alpha * mean_dct + beta * var_dct
            
            return float(niqe_score)
            
        except Exception as e:
            logger.error(f"Failed to calculate NIQE score for {image_path}: {e}")
            return None
    
    def calculate_combined_quality_score(self, image_path: Path, 
                                       brisque_weight: float = 0.5,
                                       niqe_weight: float = 0.5) -> Optional[float]:
        """
        Calculate combined quality score using BRISQUE and NIQE.
        
        Args:
            image_path: Path to the image file
            brisque_weight: Weight for BRISQUE score (0-1)
            niqe_weight: Weight for NIQE score (0-1)
            
        Returns:
            Combined quality score (lower is better) or None if failed
        """
        try:
            # Calculate individual scores
            brisque_score = self.calculate_brisque_score(image_path)
            niqe_score = self.calculate_niqe_score(image_path)
            
            if brisque_score is None or niqe_score is None:
                return None
            
            # Normalize scores to similar ranges (0-1)
            # These normalization factors would need tuning based on your dataset
            normalized_brisque = min(brisque_score / 10.0, 1.0)  # Assuming max BRISQUE ~10
            normalized_niqe = min(niqe_score / 5.0, 1.0)  # Assuming max NIQE ~5
            
            # Calculate weighted combination
            combined_score = (brisque_weight * normalized_brisque + 
                            niqe_weight * normalized_niqe)
            
            return float(combined_score)
            
        except Exception as e:
            logger.error(f"Failed to calculate combined quality score for {image_path}: {e}")
            return None
    
    def calculate_all_metrics(self, image_path: Path) -> Optional[Tuple[float, float, float]]:
        """
        Calculate all quality metrics for an image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Tuple of (brisque_score, niqe_score, combined_score) or None if failed
        """
        try:
            brisque_score = self.calculate_brisque_score(image_path)
            niqe_score = self.calculate_niqe_score(image_path)
            
            if brisque_score is None or niqe_score is None:
                return None
            
            # Calculate combined score
            combined_score = self.calculate_combined_quality_score(image_path)
            
            return (brisque_score, niqe_score, combined_score)
            
        except Exception as e:
            logger.error(f"Failed to calculate all metrics for {image_path}: {e}")
            return None
    
    def select_best_quality_image(self, image_paths: list[Path]) -> Optional[Path]:
        """
        Select the image with the best quality (least artifacts) from a list of images.
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            Path to the best quality image or None if all failed
        """
        best_image = None
        best_score = float('inf')
        
        for image_path in image_paths:
            # Use BRISQUE as primary metric since it's better at detecting artifacts
            brisque_score = self.calculate_brisque_score(image_path)
            
            if brisque_score is not None and brisque_score < best_score:
                best_score = brisque_score
                best_image = image_path
        
        return best_image 