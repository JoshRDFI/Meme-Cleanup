"""
Image utility functions for Meme-Cleanup.

Handles image loading, preprocessing, metadata extraction, and format validation.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from PIL import Image, ImageOps, ImageSequence
import numpy as np
import cv2


logger = logging.getLogger(__name__)

# Supported image formats - expanded to include all common formats
SUPPORTED_FORMATS = {
    '.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp', 
    '.gif', '.ico', '.ppm', '.pgm', '.pbm', '.pnm', '.svg',
    '.heic', '.heif', '.avif', '.jxl', '.jp2', '.j2k'
}

# Video formats to ignore
VIDEO_FORMATS = {'.mp4', '.mkv', '.avi', '.mov', '.wmv', '.flv', '.webm', '.m4v'}


def is_supported_image(file_path: Path) -> bool:
    """
    Check if a file is a supported image format.
    
    Args:
        file_path: Path to the file to check
        
    Returns:
        True if the file is a supported image format
    """
    suffix = file_path.suffix.lower()
    return suffix in SUPPORTED_FORMATS and suffix not in VIDEO_FORMATS


def is_video_file(file_path: Path) -> bool:
    """
    Check if a file is a video format.
    
    Args:
        file_path: Path to the file to check
        
    Returns:
        True if the file is a video format
    """
    return file_path.suffix.lower() in VIDEO_FORMATS


def load_image(file_path: Path, target_size: Optional[Tuple[int, int]] = None) -> Optional[np.ndarray]:
    """
    Load an image from file path, handling animated GIFs.
    
    Args:
        file_path: Path to the image file
        target_size: Optional target size (width, height) for resizing
        
    Returns:
        Loaded image as numpy array (RGB format) or None if failed
    """
    try:
        # Load with PIL for better format support
        with Image.open(file_path) as img:
            # Handle animated GIFs - take first frame
            if img.format == 'GIF' and hasattr(img, 'n_frames') and img.n_frames > 1:
                # Get first frame of animated GIF
                img.seek(0)
                logger.info(f"Processing first frame of animated GIF: {file_path}")
            
            # Convert to RGB if necessary
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Resize if target size specified
            if target_size:
                img = img.resize(target_size, Image.Resampling.LANCZOS)
            
            # Convert to numpy array
            return np.array(img)
            
    except Exception as e:
        logger.error(f"Failed to load image {file_path}: {e}")
        return None


def extract_image_metadata(file_path: Path) -> Dict[str, Any]:
    """
    Extract comprehensive metadata from an image file.
    
    Args:
        file_path: Path to the image file
        
    Returns:
        Dictionary containing image metadata
    """
    metadata = {
        'file_path': str(file_path),
        'file_size': file_path.stat().st_size,
        'file_modified': file_path.stat().st_mtime,
        'width': None,
        'height': None,
        'format': None,
        'mode': None,
        'dpi': None,
        'exif': None,
        'has_alpha': False,
        'is_animated': False,
        'frame_count': 1,
        'color_profile': None,
        'compression': None,
        'metadata_richness': 0  # Score for metadata priority
    }
    
    try:
        with Image.open(file_path) as img:
            metadata.update({
                'width': img.width,
                'height': img.height,
                'format': img.format,
                'mode': img.mode,
                'dpi': img.info.get('dpi'),
                'has_alpha': 'A' in img.mode,
                'is_animated': hasattr(img, 'n_frames') and img.n_frames > 1,
                'frame_count': getattr(img, 'n_frames', 1),
                'color_profile': img.info.get('icc_profile'),
                'compression': img.info.get('compression')
            })
            
            # Extract EXIF data if available
            if hasattr(img, '_getexif') and img._getexif():
                metadata['exif'] = dict(img._getexif())
            
            # Calculate metadata richness score
            metadata['metadata_richness'] = calculate_metadata_richness(metadata)
                
    except Exception as e:
        logger.error(f"Failed to extract metadata from {file_path}: {e}")
    
    return metadata


def calculate_metadata_richness(metadata: Dict[str, Any]) -> int:
    """
    Calculate a score for metadata richness to prioritize images with more metadata.
    
    Args:
        metadata: Image metadata dictionary
        
    Returns:
        Metadata richness score (higher = more metadata)
    """
    score = 0
    
    # EXIF data (most important)
    if metadata.get('exif'):
        score += 50
        # Bonus for specific EXIF fields
        exif = metadata['exif']
        if any(key in exif for key in [36867, 36868, 306, 315, 33432]):  # DateTime, Artist, Copyright
            score += 20
    
    # Color profile
    if metadata.get('color_profile'):
        score += 15
    
    # Alpha channel
    if metadata.get('has_alpha'):
        score += 10
    
    # DPI information
    if metadata.get('dpi'):
        score += 5
    
    # Compression info
    if metadata.get('compression'):
        score += 5
    
    return score


def copy_metadata(source_path: Path, target_path: Path) -> bool:
    """
    Copy metadata from source image to target image.
    
    Args:
        source_path: Path to source image with metadata
        target_path: Path to target image to receive metadata
        
    Returns:
        True if metadata was successfully copied
    """
    try:
        # Load source image
        with Image.open(source_path) as source_img:
            # Load target image
            with Image.open(target_path) as target_img:
                # Create new image with source metadata
                new_img = target_img.copy()
                
                # Copy EXIF data if available
                if hasattr(source_img, '_getexif') and source_img._getexif():
                    new_img.info['exif'] = source_img._getexif()
                
                # Copy other metadata
                for key, value in source_img.info.items():
                    if key not in ['exif']:  # exif handled separately
                        new_img.info[key] = value
                
                # Save with metadata
                new_img.save(target_path, format=target_img.format, **new_img.info)
                
                logger.info(f"Copied metadata from {source_path} to {target_path}")
                return True
                
    except Exception as e:
        logger.error(f"Failed to copy metadata from {source_path} to {target_path}: {e}")
        return False


def preprocess_image_for_clip(image: np.ndarray, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """
    Preprocess image for CLIP model input.
    
    Args:
        image: Input image as numpy array (RGB)
        target_size: Target size for CLIP (default: 224x224)
        
    Returns:
        Preprocessed image as numpy array
    """
    # Resize image
    resized = cv2.resize(image, target_size, interpolation=cv2.INTER_LANCZOS4)
    
    # Normalize to [0, 1] range
    normalized = resized.astype(np.float32) / 255.0
    
    return normalized


def calculate_image_hash(image: np.ndarray) -> str:
    """
    Calculate a simple perceptual hash for quick duplicate detection.
    
    Args:
        image: Input image as numpy array
        
    Returns:
        Hexadecimal string representing the image hash
    """
    # Resize to 8x8 for simple hash
    small = cv2.resize(image, (8, 8), interpolation=cv2.INTER_LANCZOS4)
    
    # Convert to grayscale
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    
    # Calculate mean
    mean = np.mean(gray)
    
    # Create hash: 1 if pixel > mean, 0 otherwise
    hash_bits = (gray > mean).flatten()
    
    # Convert to hex string
    hash_hex = ''.join(['1' if bit else '0' for bit in hash_bits])
    hash_int = int(hash_hex, 2)
    
    return f"{hash_int:016x}"


def get_image_files(directory: Path, recursive: bool = True) -> List[Path]:
    """
    Get all image files from a directory, excluding video files.
    
    Args:
        directory: Directory to scan
        recursive: Whether to scan subdirectories
        
    Returns:
        List of image file paths
    """
    image_files = []
    
    if recursive:
        pattern = "**/*"
    else:
        pattern = "*"
    
    for file_path in directory.glob(pattern):
        if file_path.is_file() and is_supported_image(file_path):
            image_files.append(file_path)
    
    return sorted(image_files)


def get_all_media_files(directory: Path, recursive: bool = True) -> Tuple[List[Path], List[Path]]:
    """
    Get all media files from a directory, separating images and videos.
    
    Args:
        directory: Directory to scan
        recursive: Whether to scan subdirectories
        
    Returns:
        Tuple of (image_files, video_files)
    """
    image_files = []
    video_files = []
    
    if recursive:
        pattern = "**/*"
    else:
        pattern = "*"
    
    for file_path in directory.glob(pattern):
        if file_path.is_file():
            if is_supported_image(file_path):
                image_files.append(file_path)
            elif is_video_file(file_path):
                video_files.append(file_path)
    
    return sorted(image_files), sorted(video_files) 