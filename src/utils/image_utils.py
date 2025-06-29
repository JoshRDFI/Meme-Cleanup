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
            # Basic image properties
            metadata['width'] = img.width
            metadata['height'] = img.height
            metadata['format'] = img.format
            metadata['mode'] = img.mode
            
            # DPI information
            if hasattr(img, 'info') and 'dpi' in img.info:
                metadata['dpi'] = img.info['dpi']
            
            # Alpha channel
            if img.mode in ('RGBA', 'LA', 'PA'):
                metadata['has_alpha'] = True
            
            # Animation info
            if hasattr(img, 'n_frames') and img.n_frames > 1:
                metadata['is_animated'] = True
                metadata['frame_count'] = img.n_frames
            
            # EXIF data
            if hasattr(img, '_getexif') and img._getexif():
                metadata['exif'] = dict(img._getexif())
            
            # Color profile
            if hasattr(img, 'info') and 'icc_profile' in img.info:
                metadata['color_profile'] = 'ICC'
            
            # Compression info
            if hasattr(img, 'info') and 'compression' in img.info:
                metadata['compression'] = img.info['compression']
            
            # Calculate metadata richness score
            metadata['metadata_richness'] = calculate_metadata_richness(metadata)
            
    except Exception as e:
        logger.error(f"Failed to extract metadata from {file_path}: {e}")
    
    return metadata


def calculate_metadata_richness(metadata: Dict[str, Any]) -> int:
    """
    Calculate a score for metadata richness to help prioritize images.
    
    Args:
        metadata: Image metadata dictionary
        
    Returns:
        Richness score (higher = more metadata)
    """
    score = 0
    
    # Basic properties
    if metadata.get('width') and metadata.get('height'):
        score += 1
    
    if metadata.get('format'):
        score += 1
    
    # DPI information
    if metadata.get('dpi'):
        score += 2
    
    # EXIF data
    if metadata.get('exif'):
        score += 3
    
    # Color profile
    if metadata.get('color_profile'):
        score += 2
    
    # Compression info
    if metadata.get('compression'):
        score += 1
    
    # Animation (negative score as we prefer static images)
    if metadata.get('is_animated'):
        score -= 1
    
    return score


def copy_metadata(source_path: Path, target_path: Path) -> bool:
    """
    Copy metadata from source image to target image.
    
    Args:
        source_path: Path to source image
        target_path: Path to target image
        
    Returns:
        True if successful, False otherwise
    """
    try:
        with Image.open(source_path) as source_img:
            with Image.open(target_path) as target_img:
                # Copy EXIF data if available
                if hasattr(source_img, '_getexif') and source_img._getexif():
                    exif_data = source_img._getexif()
                    target_img.save(target_path, exif=exif_data)
                    return True
    except Exception as e:
        logger.error(f"Failed to copy metadata from {source_path} to {target_path}: {e}")
    
    return False


def preprocess_image_for_clip(image: np.ndarray, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """
    Preprocess image for CLIP model input.
    
    Args:
        image: Input image as numpy array
        target_size: Target size (width, height)
        
    Returns:
        Preprocessed image as numpy array
    """
    try:
        # Convert to PIL Image
        pil_image = Image.fromarray(image)
        
        # Resize
        pil_image = pil_image.resize(target_size, Image.Resampling.LANCZOS)
        
        # Convert back to numpy array
        return np.array(pil_image)
        
    except Exception as e:
        logger.error(f"Failed to preprocess image for CLIP: {e}")
        return image


def calculate_image_hash(image: np.ndarray) -> str:
    """
    Calculate perceptual hash of an image for quick duplicate detection.
    
    Args:
        image: Input image as numpy array
        
    Returns:
        Perceptual hash string
    """
    try:
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Resize to 8x8
        resized = cv2.resize(gray, (8, 8))
        
        # Calculate mean
        mean = resized.mean()
        
        # Create hash
        hash_str = ''
        for row in resized:
            for pixel in row:
                hash_str += '1' if pixel > mean else '0'
        
        return hash_str
        
    except Exception as e:
        logger.error(f"Failed to calculate image hash: {e}")
        return ''


def get_valid_image_files(directory: Path, recursive: bool = True) -> List[Path]:
    """
    Get all valid image files from a directory.
    
    Args:
        directory: Directory to scan
        recursive: Whether to scan subdirectories
        
    Returns:
        List of valid image file paths
    """
    image_files = []
    
    if not directory.exists():
        logger.warning(f"Directory does not exist: {directory}")
        return image_files
    
    # Get all files
    if recursive:
        all_files = list(directory.rglob('*'))
    else:
        all_files = list(directory.iterdir())
    
    # Filter for valid image files
    for file_path in all_files:
        if file_path.is_file() and is_supported_image(file_path):
            if is_valid_image_file(file_path):
                image_files.append(file_path)
            else:
                logger.warning(f"Skipping corrupted image: {file_path}")
    
    return image_files


def get_all_media_files(directory: Path, recursive: bool = True) -> Tuple[List[Path], List[Path], List[Path]]:
    """
    Get all media files from a directory, categorized by type.
    
    Args:
        directory: Directory to scan
        recursive: Whether to scan subdirectories
        
    Returns:
        Tuple of (valid_images, videos, corrupted_files)
    """
    valid_images = []
    videos = []
    corrupted = []
    
    if not directory.exists():
        logger.warning(f"Directory does not exist: {directory}")
        return valid_images, videos, corrupted
    
    # Get all files
    if recursive:
        all_files = list(directory.rglob('*'))
    else:
        all_files = list(directory.iterdir())
    
    # Categorize files
    for file_path in all_files:
        if not file_path.is_file():
            continue
        
        if is_video_file(file_path):
            videos.append(file_path)
        elif is_supported_image(file_path):
            if is_valid_image_file(file_path):
                valid_images.append(file_path)
            else:
                corrupted.append(file_path)
    
    return valid_images, videos, corrupted


def get_scan_summary(scan_results: Dict[str, Any]) -> str:
    """
    Generate a summary string from scan results.
    
    Args:
        scan_results: Dictionary containing scan statistics
        
    Returns:
        Formatted summary string
    """
    summary = f"""
Scan Summary:
=============
Directories Scanned: {scan_results.get('directories_scanned', 0)}
Total Images Found: {scan_results.get('total_images_found', 0)}
Total Images Processed: {scan_results.get('total_images_processed', 0)}
Corrupted Files Skipped: {scan_results.get('corrupted_files_skipped', 0)}
Skipped Files: {scan_results.get('skipped_files_count', 0)}
Scan Duration: {scan_results.get('scan_duration', 0):.2f} seconds
"""
    return summary


def test_image_file(file_path: Path) -> Dict[str, Any]:
    """
    Comprehensive test of an image file for debugging.
    
    Args:
        file_path: Path to the image file to test
        
    Returns:
        Dictionary with test results
    """
    result = {
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
        result['validation_steps'].append("File exists")
        
        # Check file size
        result['file_size'] = file_path.stat().st_size
        if result['file_size'] == 0:
            result['error'] = "File is empty"
            return result
        
        result['validation_steps'].append(f"File size: {result['file_size']} bytes")
        
        # Try to open with PIL
        with Image.open(file_path) as img:
            result['validation_steps'].append("PIL can open file")
            
            # Get basic properties
            result['format'] = img.format
            result['dimensions'] = (img.width, img.height)
            result['validation_steps'].append(f"Format: {img.format}, Size: {img.width}x{img.height}")
            
            # Try to convert to RGB
            rgb_img = img.convert('RGB')
            result['validation_steps'].append("Can convert to RGB")
            
            # Try to get numpy array
            array = np.array(rgb_img)
            result['validation_steps'].append(f"Can create numpy array: {array.shape}")
            
            # Success
            result['is_valid'] = True
            result['validation_steps'].append("Image is valid")
            
    except Exception as e:
        result['error'] = str(e)
        result['validation_steps'].append(f"Error: {e}")
    
    return result 