<<<<<<< HEAD
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


def is_valid_image_file(file_path: Path, debug: bool = False) -> bool:
    """
    Check if a file is actually a valid image file by attempting to open it.
    Uses a lenient approach to avoid false positives.
    
    Args:
        file_path: Path to the file to check
        debug: If True, log detailed validation steps
        
    Returns:
        True if the file is a valid image that can be opened
    """
    try:
        # First check if file exists and has content
        if not file_path.exists():
            if debug:
                logger.debug(f"File does not exist: {file_path}")
            return False
            
        if file_path.stat().st_size == 0:
            if debug:
                logger.debug(f"File is empty: {file_path}")
            return False
        
        if debug:
            logger.debug(f"Validating image: {file_path}")
        
        # Try to open the image with PIL
        with Image.open(file_path) as img:
            # Try to access basic properties - this is much more lenient than verify()
            try:
                # Just try to get basic info - don't use verify() which is too strict
                width, height = img.size
                format_name = img.format
                
                if debug:
                    logger.debug(f"  Format: {format_name}, Size: {width}x{height}")
                
                # Basic sanity checks
                if width <= 0 or height <= 0:
                    logger.warning(f"Invalid dimensions in {file_path}: {width}x{height}")
                    return False
                
                if not format_name:
                    logger.warning(f"No format detected for {file_path}")
                    return False
                
                # Try to load the first frame (for animated images)
                if hasattr(img, 'n_frames') and img.n_frames > 1:
                    if debug:
                        logger.debug(f"  Animated image with {img.n_frames} frames")
                    # For animated images, try to load the first frame
                    img.seek(0)
                
                # Try to convert to RGB to test if the image can be processed
                # This catches most corruption issues without being too strict
                test_img = img.convert('RGB')
                test_array = np.array(test_img)
                
                # Basic array sanity check
                if test_array.size == 0:
                    logger.warning(f"Empty image array for {file_path}")
                    return False
                
                if debug:
                    logger.debug(f"  Validation successful for {file_path}")
                return True
                
            except (OSError, ValueError, TypeError) as e:
                # These are the errors that indicate actual corruption
                logger.warning(f"Image corruption detected in {file_path}: {e}")
                return False
                
    except Exception as e:
        # Log the specific error for debugging
        logger.warning(f"Failed to validate {file_path}: {e}")
        return False


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
            
            # Extract EXIF data if available - convert to serializable format
            try:
                if hasattr(img, '_getexif') and img._getexif():
                    exif_data = img._getexif()
                    # Convert EXIF data to serializable format
                    serializable_exif = {}
                    for tag_id, value in exif_data.items():
                        try:
                            # Convert IFDRational and other PIL types to simple values
                            if hasattr(value, 'numerator') and hasattr(value, 'denominator'):
                                # Handle IFDRational
                                serializable_exif[tag_id] = float(value.numerator) / float(value.denominator)
                            elif isinstance(value, (int, float, str, bool)):
                                serializable_exif[tag_id] = value
                            else:
                                # Convert other types to string
                                serializable_exif[tag_id] = str(value)
                        except Exception as e:
                            logger.warning(f"Failed to serialize EXIF tag {tag_id}: {e}")
                            # Skip problematic EXIF tags
                            continue
                    
                    metadata['exif'] = serializable_exif
            except Exception as e:
                logger.warning(f"Failed to extract EXIF data from {file_path}: {e}")
                # Continue without EXIF data
                metadata['exif'] = None
            
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


def get_valid_image_files(directory: Path, recursive: bool = True) -> List[Path]:
    """
    Get all valid image files from a directory, filtering out corrupted files.
    
    Args:
        directory: Directory to scan
        recursive: Whether to scan subdirectories
        
    Returns:
        List of valid image file paths
    """
    image_files = []
    corrupted_files = []
    
    if recursive:
        pattern = "**/*"
    else:
        pattern = "*"
    
    for file_path in directory.glob(pattern):
        if file_path.is_file() and is_supported_image(file_path):
            if is_valid_image_file(file_path):
                image_files.append(file_path)
            else:
                corrupted_files.append(file_path)
                logger.warning(f"Corrupted or invalid image file: {file_path}")
    
    if corrupted_files:
        logger.warning(f"Found {len(corrupted_files)} corrupted/invalid image files in {directory}")
    
    return sorted(image_files)


def get_all_media_files(directory: Path, recursive: bool = True) -> Tuple[List[Path], List[Path], List[Path]]:
    """
    Get all media files from a directory, separating images, videos, and corrupted files.
    
    Args:
        directory: Directory to scan
        recursive: Whether to scan subdirectories
        
    Returns:
        Tuple of (valid_image_files, video_files, corrupted_files)
    """
    image_files = []
    video_files = []
    corrupted_files = []
    
    if recursive:
        pattern = "**/*"
    else:
        pattern = "*"
    
    for file_path in directory.glob(pattern):
        if file_path.is_file():
            if is_supported_image(file_path):
                if is_valid_image_file(file_path):
                    image_files.append(file_path)
                else:
                    corrupted_files.append(file_path)
            elif is_video_file(file_path):
                video_files.append(file_path)
    
    return sorted(image_files), sorted(video_files), sorted(corrupted_files)


def get_scan_summary(directory: Path, recursive: bool = True) -> Dict[str, Any]:
    """
    Get a summary of all files in a directory.
    
    Args:
        directory: Directory to scan
        recursive: Whether to scan subdirectories
        
    Returns:
        Dictionary with scan summary
    """
    valid_images, videos, corrupted = get_all_media_files(directory, recursive)
    
    return {
        'directory': str(directory),
        'valid_images': len(valid_images),
        'video_files': len(videos),
        'corrupted_files': len(corrupted),
        'total_files': len(valid_images) + len(videos) + len(corrupted),
        'corrupted_file_paths': [str(f) for f in corrupted[:10]]  # First 10 for reporting
    }


def test_image_file(file_path: Path) -> Dict[str, Any]:
    """
    Test a specific image file and return detailed validation results.
    Useful for debugging validation issues.
    
    Args:
        file_path: Path to the image file to test
        
    Returns:
        Dictionary with detailed test results
    """
    result = {
        'file_path': str(file_path),
        'exists': False,
        'file_size': 0,
        'is_valid': False,
        'format': None,
        'dimensions': None,
        'error': None,
        'validation_steps': []
    }
    
    try:
        # Check if file exists
        if not file_path.exists():
            result['error'] = "File does not exist"
            return result
        
        result['exists'] = True
        result['file_size'] = file_path.stat().st_size
        
        if result['file_size'] == 0:
            result['error'] = "File is empty"
            return result
        
        result['validation_steps'].append("File exists and has content")
        
        # Try to open with PIL
        with Image.open(file_path) as img:
            result['validation_steps'].append("Successfully opened with PIL")
            
            # Get basic info
            result['format'] = img.format
            result['dimensions'] = (img.width, img.height)
            
            result['validation_steps'].append(f"Format: {img.format}, Size: {img.width}x{img.height}")
            
            # Check for animation
            if hasattr(img, 'n_frames') and img.n_frames > 1:
                result['validation_steps'].append(f"Animated image with {img.n_frames} frames")
                img.seek(0)
            
            # Try to convert to RGB
            test_img = img.convert('RGB')
            result['validation_steps'].append("Successfully converted to RGB")
            
            # Try to convert to numpy array
            test_array = np.array(test_img)
            result['validation_steps'].append(f"Successfully converted to numpy array: {test_array.shape}")
            
            result['is_valid'] = True
            
    except Exception as e:
        result['error'] = str(e)
        result['validation_steps'].append(f"Error: {e}")
    
=======
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


def is_valid_image_file(file_path: Path, debug: bool = False) -> bool:
    """
    Check if a file is actually a valid image file by attempting to open it.
    Uses a lenient approach to avoid false positives.
    
    Args:
        file_path: Path to the file to check
        debug: If True, log detailed validation steps
        
    Returns:
        True if the file is a valid image that can be opened
    """
    try:
        # First check if file exists and has content
        if not file_path.exists():
            if debug:
                logger.debug(f"File does not exist: {file_path}")
            return False
            
        if file_path.stat().st_size == 0:
            if debug:
                logger.debug(f"File is empty: {file_path}")
            return False
        
        if debug:
            logger.debug(f"Validating image: {file_path}")
        
        # Try to open the image with PIL
        with Image.open(file_path) as img:
            # Try to access basic properties - this is much more lenient than verify()
            try:
                # Just try to get basic info - don't use verify() which is too strict
                width, height = img.size
                format_name = img.format
                
                if debug:
                    logger.debug(f"  Format: {format_name}, Size: {width}x{height}")
                
                # Basic sanity checks
                if width <= 0 or height <= 0:
                    logger.warning(f"Invalid dimensions in {file_path}: {width}x{height}")
                    return False
                
                if not format_name:
                    logger.warning(f"No format detected for {file_path}")
                    return False
                
                # Try to load the first frame (for animated images)
                if hasattr(img, 'n_frames') and img.n_frames > 1:
                    if debug:
                        logger.debug(f"  Animated image with {img.n_frames} frames")
                    # For animated images, try to load the first frame
                    img.seek(0)
                
                # Try to convert to RGB to test if the image can be processed
                # This catches most corruption issues without being too strict
                test_img = img.convert('RGB')
                test_array = np.array(test_img)
                
                # Basic array sanity check
                if test_array.size == 0:
                    logger.warning(f"Empty image array for {file_path}")
                    return False
                
                if debug:
                    logger.debug(f"  Validation successful for {file_path}")
                return True
                
            except (OSError, ValueError, TypeError) as e:
                # These are the errors that indicate actual corruption
                logger.warning(f"Image corruption detected in {file_path}: {e}")
                return False
                
    except Exception as e:
        # Log the specific error for debugging
        logger.warning(f"Failed to validate {file_path}: {e}")
        return False


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
            
            # Extract EXIF data if available - convert to serializable format
            try:
                if hasattr(img, '_getexif') and img._getexif():
                    exif_data = img._getexif()
                    # Convert EXIF data to serializable format
                    serializable_exif = {}
                    for tag_id, value in exif_data.items():
                        try:
                            # Convert IFDRational and other PIL types to simple values
                            if hasattr(value, 'numerator') and hasattr(value, 'denominator'):
                                # Handle IFDRational
                                serializable_exif[tag_id] = float(value.numerator) / float(value.denominator)
                            elif isinstance(value, (int, float, str, bool)):
                                serializable_exif[tag_id] = value
                            else:
                                # Convert other types to string
                                serializable_exif[tag_id] = str(value)
                        except Exception as e:
                            logger.warning(f"Failed to serialize EXIF tag {tag_id}: {e}")
                            # Skip problematic EXIF tags
                            continue
                    
                    metadata['exif'] = serializable_exif
            except Exception as e:
                logger.warning(f"Failed to extract EXIF data from {file_path}: {e}")
                # Continue without EXIF data
                metadata['exif'] = None
            
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


def get_valid_image_files(directory: Path, recursive: bool = True) -> List[Path]:
    """
    Get all valid image files from a directory, filtering out corrupted files.
    
    Args:
        directory: Directory to scan
        recursive: Whether to scan subdirectories
        
    Returns:
        List of valid image file paths
    """
    image_files = []
    corrupted_files = []
    
    if recursive:
        pattern = "**/*"
    else:
        pattern = "*"
    
    for file_path in directory.glob(pattern):
        if file_path.is_file() and is_supported_image(file_path):
            if is_valid_image_file(file_path):
                image_files.append(file_path)
            else:
                corrupted_files.append(file_path)
                logger.warning(f"Corrupted or invalid image file: {file_path}")
    
    if corrupted_files:
        logger.warning(f"Found {len(corrupted_files)} corrupted/invalid image files in {directory}")
    
    return sorted(image_files)


def get_all_media_files(directory: Path, recursive: bool = True) -> Tuple[List[Path], List[Path], List[Path]]:
    """
    Get all media files from a directory, separating images, videos, and corrupted files.
    
    Args:
        directory: Directory to scan
        recursive: Whether to scan subdirectories
        
    Returns:
        Tuple of (valid_image_files, video_files, corrupted_files)
    """
    image_files = []
    video_files = []
    corrupted_files = []
    
    if recursive:
        pattern = "**/*"
    else:
        pattern = "*"
    
    for file_path in directory.glob(pattern):
        if file_path.is_file():
            if is_supported_image(file_path):
                if is_valid_image_file(file_path):
                    image_files.append(file_path)
                else:
                    corrupted_files.append(file_path)
            elif is_video_file(file_path):
                video_files.append(file_path)
    
    return sorted(image_files), sorted(video_files), sorted(corrupted_files)


def get_scan_summary(directory: Path, recursive: bool = True) -> Dict[str, Any]:
    """
    Get a summary of all files in a directory.
    
    Args:
        directory: Directory to scan
        recursive: Whether to scan subdirectories
        
    Returns:
        Dictionary with scan summary
    """
    valid_images, videos, corrupted = get_all_media_files(directory, recursive)
    
    return {
        'directory': str(directory),
        'valid_images': len(valid_images),
        'video_files': len(videos),
        'corrupted_files': len(corrupted),
        'total_files': len(valid_images) + len(videos) + len(corrupted),
        'corrupted_file_paths': [str(f) for f in corrupted[:10]]  # First 10 for reporting
    }


def test_image_file(file_path: Path) -> Dict[str, Any]:
    """
    Test a specific image file and return detailed validation results.
    Useful for debugging validation issues.
    
    Args:
        file_path: Path to the image file to test
        
    Returns:
        Dictionary with detailed test results
    """
    result = {
        'file_path': str(file_path),
        'exists': False,
        'file_size': 0,
        'is_valid': False,
        'format': None,
        'dimensions': None,
        'error': None,
        'validation_steps': []
    }
    
    try:
        # Check if file exists
        if not file_path.exists():
            result['error'] = "File does not exist"
            return result
        
        result['exists'] = True
        result['file_size'] = file_path.stat().st_size
        
        if result['file_size'] == 0:
            result['error'] = "File is empty"
            return result
        
        result['validation_steps'].append("File exists and has content")
        
        # Try to open with PIL
        with Image.open(file_path) as img:
            result['validation_steps'].append("Successfully opened with PIL")
            
            # Get basic info
            result['format'] = img.format
            result['dimensions'] = (img.width, img.height)
            
            result['validation_steps'].append(f"Format: {img.format}, Size: {img.width}x{img.height}")
            
            # Check for animation
            if hasattr(img, 'n_frames') and img.n_frames > 1:
                result['validation_steps'].append(f"Animated image with {img.n_frames} frames")
                img.seek(0)
            
            # Try to convert to RGB
            test_img = img.convert('RGB')
            result['validation_steps'].append("Successfully converted to RGB")
            
            # Try to convert to numpy array
            test_array = np.array(test_img)
            result['validation_steps'].append(f"Successfully converted to numpy array: {test_array.shape}")
            
            result['is_valid'] = True
            
    except Exception as e:
        result['error'] = str(e)
        result['validation_steps'].append(f"Error: {e}")
    
>>>>>>> 88a757b8ecc4ae4d819ae02a34c56b2a1ebc3714
    return result 