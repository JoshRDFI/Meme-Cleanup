"""
Tests for image utility functions.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch

from src.utils.image_utils import (
    is_supported_image, get_image_files, calculate_image_hash
)


class TestImageUtils:
    """Test cases for image utility functions."""
    
    def test_is_supported_image(self):
        """Test supported image format detection."""
        # Test supported formats
        assert is_supported_image(Path("test.jpg"))
        assert is_supported_image(Path("test.jpeg"))
        assert is_supported_image(Path("test.png"))
        assert is_supported_image(Path("test.bmp"))
        assert is_supported_image(Path("test.tiff"))
        assert is_supported_image(Path("test.webp"))
        
        # Test unsupported formats
        assert not is_supported_image(Path("test.txt"))
        assert not is_supported_image(Path("test.pdf"))
        assert not is_supported_image(Path("test.mp4"))
    
    def test_get_image_files(self, tmp_path):
        """Test image file discovery."""
        # Create test files
        (tmp_path / "test.jpg").touch()
        (tmp_path / "test.png").touch()
        (tmp_path / "test.txt").touch()
        (tmp_path / "subdir").mkdir()
        (tmp_path / "subdir" / "test.bmp").touch()
        
        # Test non-recursive
        files = get_image_files(tmp_path, recursive=False)
        assert len(files) == 2
        assert any("test.jpg" in str(f) for f in files)
        assert any("test.png" in str(f) for f in files)
        
        # Test recursive
        files = get_image_files(tmp_path, recursive=True)
        assert len(files) == 3
        assert any("test.jpg" in str(f) for f in files)
        assert any("test.png" in str(f) for f in files)
        assert any("test.bmp" in str(f) for f in files)
    
    @patch('src.utils.image_utils.cv2.resize')
    @patch('src.utils.image_utils.cv2.cvtColor')
    def test_calculate_image_hash(self, mock_cvt_color, mock_resize):
        """Test image hash calculation."""
        # Mock numpy array
        import numpy as np
        mock_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        # Mock OpenCV functions
        mock_resize.return_value = np.random.randint(0, 255, (8, 8), dtype=np.uint8)
        mock_cvt_color.return_value = np.random.randint(0, 255, (8, 8), dtype=np.uint8)
        
        # Test hash calculation
        hash_result = calculate_image_hash(mock_image)
        assert isinstance(hash_result, str)
        assert len(hash_result) == 16  # 64 bits = 16 hex chars 