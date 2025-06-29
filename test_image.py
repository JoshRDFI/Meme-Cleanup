#!/usr/bin/env python3
"""
Simple script to test image validation for debugging purposes.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from utils.image_utils import test_image_file, is_valid_image_file

def main():
    if len(sys.argv) < 2:
        print("Usage: python test_image.py <image_file_path>")
        print("Example: python test_image.py E:\\images\\subdir\\image_name.jpg
    
    file_path = Path(sys.argv[1])
    
    print(f"Testing image file: {file_path}")
    print("=" * 50)
    
    # Test with detailed validation
    result = test_image_file(file_path)
    
    print(f"File exists: {result['exists']}")
    print(f"File size: {result['file_size']} bytes")
    print(f"Is valid: {result['is_valid']}")
    print(f"Format: {result['format']}")
    print(f"Dimensions: {result['dimensions']}")
    
    if result['error']:
        print(f"Error: {result['error']}")
    
    print("\nValidation steps:")
    for step in result['validation_steps']:
        print(f"  ✓ {step}")
    
    print("\n" + "=" * 50)
    
    # Test with the validation function
    is_valid = is_valid_image_file(file_path, debug=True)
    print(f"is_valid_image_file() result: {is_valid}")
    
    if is_valid:
        print("✅ Image is considered valid by the app")
    else:
        print("❌ Image is considered corrupted by the app")

if __name__ == "__main__":
    main() 