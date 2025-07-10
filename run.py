#!/usr/bin/env python3
"""
Launcher script for Meme-Cleanup.

This script provides a simple way to run the application and handles
basic setup and error checking.
"""

import sys
import os
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 12):
        print("Error: Python 3.12 or higher is required.")
        print(f"Current version: {sys.version}")
        sys.exit(1)

def check_dependencies():
    """Check if required dependencies are available."""
    try:
        import PyQt6
        import torch
        import cv2
        import numpy
        import PIL
        import transformers
        print("✓ All required dependencies are available")
    except ImportError as e:
        print(f"Error: Missing dependency - {e}")
        print("Please install dependencies with: pip install -r requirements.txt")
        sys.exit(1)

def setup_environment():
    """Setup environment variables for optimal performance."""
    # Disable symlinks warning on Windows
    os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
    
    # Set PyTorch to use deterministic algorithms for reproducibility
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    
    print("✓ Environment configured for optimal performance")

def main():
    """Main launcher function."""
    print("Meme-Cleanup Launcher")
    print("=" * 50)
    
    # Check Python version
    check_python_version()
    
    # Setup environment
    setup_environment()
    
    # Check dependencies
    check_dependencies()
    
    # Add src directory to path
    src_path = Path(__file__).parent / "src"
    if src_path.exists():
        sys.path.insert(0, str(src_path))
    else:
        print("Error: src directory not found")
        sys.exit(1)
    
    # Import and run main application
    try:
        import main
        print("Starting Meme-Cleanup...")
        main.main()
    except Exception as e:
        print(f"Error starting application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 