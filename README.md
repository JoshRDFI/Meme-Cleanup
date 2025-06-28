# Meme-Cleanup

A high-performance, GPU-accelerated image deduplication tool with a modern PyQt6 interface. Built for organizing large meme collections and image libraries using AI-powered similarity detection.

## 🚀 Features

- **AI-Powered Detection**: Uses CLIP embeddings for intelligent visual similarity detection
- **Quality Assessment**: BRISQUE and NIQE metrics for selecting the best quality images
- **GPU Acceleration**: CUDA support for fast processing of large image collections
- **Modern UI**: Dark-themed PyQt6 interface with real-time progress tracking
- **Batch Processing**: Efficient handling of thousands of images
- **Database Storage**: SQLite backend for metadata and progress persistence
- **Cross-Platform**: Works on Windows, macOS, and Linux

## 📋 Requirements

### System Requirements
- **OS**: Windows 10+, macOS 10.15+, or Linux
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 2GB for application + space for image database
- **GPU**: NVIDIA RTX 5080+ with CUDA support (PyTorch nightly builds required)
- **CPU**: Multi-core processor for parallel processing

### Python Requirements
- **Python**: 3.12 or higher
- **PyTorch**: 2.2.0+ nightly builds (required for RTX 5080 support)
- **PyQt6**: 6.5.0+ for the GUI
- **OpenCV**: 4.8.0+ for image processing
- **Transformers**: 4.35.0+ for CLIP model
- **Joblib**: 1.3.0+ for parallel processing

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/meme-cleanup.git
cd meme-cleanup
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Application
```bash
python src/main.py
```

## 🎯 Quick Start

1. **Launch the Application**
   - Run `python src/main.py`
   - The application will open with a dark-themed interface

2. **Add Source Directories**
   - Click "Add Directory" in the Scan tab
   - Select folders containing your images
   - Support for JPEG, PNG, BMP, TIFF, and WebP formats

3. **Configure Processing**
   - Set similarity threshold (0.1-1.0, default: 0.8)
   - Choose batch size (1-32, default: 8)
   - Enable/disable GPU acceleration
   - Select quality metric (combined, BRISQUE, or NIQE)

4. **Start Processing**
   - Click "Start Processing" to begin deduplication
   - Monitor progress in real-time
   - Processing includes:
     - Directory scanning and metadata extraction
     - CLIP embedding generation
     - Quality metric calculation
     - Duplicate detection and grouping

5. **Review Results**
   - Switch to the "Review Duplicates" tab
   - Browse duplicate groups
   - Select which images to keep
   - Use "Auto-Select Best" for automatic quality-based selection

## 📁 Project Structure

```
Meme-Cleanup/
├── src/
│   ├── main.py                 # Application entry point
│   ├── gui/                    # PyQt6 user interface
│   │   ├── main_window.py      # Main application window
│   │   ├── styles.py           # Dark theme styling
│   │   └── tabs/               # Tab components
│   │       ├── scan_tab.py     # Directory scanning
│   │       ├── review_tab.py   # Duplicate review
│   │       ├── settings_tab.py # Configuration
│   │       └── logs_tab.py     # Application logs
│   ├── core/                   # Core processing logic
│   │   ├── deduplicator.py     # Main deduplication engine
│   │   ├── clip_processor.py   # CLIP model integration
│   │   └── quality_metrics.py  # Quality assessment
│   ├── db/                     # Database management
│   │   └── database.py         # SQLite operations
│   ├── utils/                  # Utility functions
│   │   ├── image_utils.py      # Image processing
│   │   └── logging_config.py   # Logging setup
│   └── modules/                # Optional extensions
├── tests/                      # Unit and integration tests
├── _docs/                      # Documentation
│   └── phases/                 # Development phases
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## 🔧 Configuration

### Processing Settings
- **Similarity Threshold**: Controls how similar images must be to be considered duplicates
- **Batch Size**: Number of images processed simultaneously (affects memory usage)
- **Quality Metric**: Choose between combined, BRISQUE, or NIQE scoring
- **GPU Usage**: Enable/disable CUDA acceleration

### Database Location
- Default: `~/.meme_cleanup/meme_cleanup.db`
- Can be changed in Settings tab

## 🎨 User Interface

### Dark Theme
The application features a modern dark theme with:
- Primary color: #4A90E2 (blue)
- Background: #1E1E1E (dark gray)
- Text: #F5F5F5 (light gray)
- Accents: #E94E77 (red for warnings/errors)

### Tabs
1. **Scan & Process**: Directory selection and processing
2. **Review Duplicates**: Browse and manage duplicate groups
3. **Settings**: Application configuration
4. **Logs**: Real-time application logs

## 🚀 Performance

### Processing Speed
- **GPU (CUDA)**: ~100 images/second
- **CPU**: ~10-20 images/second
- **Memory Usage**: ~2GB for CLIP model + batch processing

### Scalability
- Tested with 10,000+ images
- Efficient database indexing
- Batch processing for large collections

## 🧪 Testing

Run the test suite:
```bash
pytest tests/
```

Current test coverage includes:
- Image utility functions
- Database operations
- CLIP processing
- Quality metrics

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

### Development Setup
```bash
pip install -r requirements.txt
pip install pytest black flake8 mypy
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **CLIP**: OpenAI's Contrastive Language-Image Pre-training model
- **PyQt6**: Modern Python bindings for Qt
- **PyTorch**: Deep learning framework
- **OpenCV**: Computer vision library

## 🐛 Known Issues

- Large image collections (>50,000 images) may require significant memory
- GPU memory usage scales with batch size
- Some image formats may not preserve all metadata

## 📞 Support

For issues and questions:
1. Check the [Issues](https://github.com/yourusername/meme-cleanup/issues) page
2. Review the logs in the application
3. Create a new issue with detailed information

## 🔄 Roadmap

### Phase 3: Enhanced Duplicate Detection
- Face recognition for portrait deduplication
- Advanced similarity algorithms
- Custom similarity thresholds per image type

### Phase 4: File Operations
- Safe file deletion with backup
- Batch export functionality
- Metadata preservation

### Phase 5: Advanced Features
- Image enhancement
- Batch renaming
- Cloud storage integration
- Plugin system for custom algorithms
