"""
Scan tab for Meme-Cleanup.

Handles directory scanning, image processing, and progress tracking.
"""

import logging
from pathlib import Path
from typing import List, Optional

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QListWidget,
    QFileDialog, QProgressBar, QGroupBox, QCheckBox, QSpinBox, QDoubleSpinBox,
    QTextEdit, QMessageBox, QFrame, QComboBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QFont

from db.database import DatabaseManager
from core.deduplicator import Deduplicator, DeduplicationConfig


logger = logging.getLogger(__name__)


class ProcessingThread(QThread):
    """Thread for running deduplication processing."""
    
    progress_updated = pyqtSignal(int, int, str)  # value, maximum, message
    processing_finished = pyqtSignal(list)  # duplicate groups
    error_occurred = pyqtSignal(str)  # error message
    
    def __init__(self, deduplicator: Deduplicator, source_directories: List[Path]):
        super().__init__()
        self.deduplicator = deduplicator
        self.source_directories = source_directories
        self._stop_requested = False
    
    def run(self):
        """Run the deduplication process."""
        try:
            # Run full deduplication
            duplicate_groups = self.deduplicator.run_full_deduplication(self.source_directories)
            self.processing_finished.emit(duplicate_groups)
        except Exception as e:
            logger.error(f"Processing failed: {e}")
            self.error_occurred.emit(str(e))
    
    def stop(self):
        """Request stop of processing."""
        self._stop_requested = True


class ScanTab(QWidget):
    """Tab for scanning directories and processing images."""
    
    progress_updated = pyqtSignal(int, int, str)
    processing_started = pyqtSignal()
    processing_finished = pyqtSignal()
    
    def __init__(self, db_manager: DatabaseManager):
        """
        Initialize scan tab.
        
        Args:
            db_manager: Database manager instance
        """
        super().__init__()
        self.db_manager = db_manager
        self.deduplicator = None
        self.processing_thread = None
        self.source_directories = []
        
        self.setup_ui()
        logger.info("Scan tab initialized")
    
    def setup_ui(self):
        """Setup the user interface."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        
        # Title
        title_label = QLabel("Scan & Process Images")
        title_font = QFont()
        title_font.setPointSize(16)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setStyleSheet("color: #4A90E2; margin-bottom: 10px;")
        layout.addWidget(title_label)
        
        # Directory selection group
        dir_group = QGroupBox("Source Directories")
        dir_layout = QVBoxLayout(dir_group)
        
        # Directory list
        self.dir_list = QListWidget()
        self.dir_list.setMaximumHeight(150)
        dir_layout.addWidget(self.dir_list)
        
        # Directory buttons
        dir_buttons_layout = QHBoxLayout()
        
        self.add_dir_button = QPushButton("Add Directory")
        self.add_dir_button.clicked.connect(self.add_directory)
        dir_buttons_layout.addWidget(self.add_dir_button)
        
        self.remove_dir_button = QPushButton("Remove Directory")
        self.remove_dir_button.clicked.connect(self.remove_directory)
        dir_buttons_layout.addWidget(self.remove_dir_button)
        
        self.clear_dirs_button = QPushButton("Clear All")
        self.clear_dirs_button.clicked.connect(self.clear_directories)
        dir_buttons_layout.addWidget(self.clear_dirs_button)
        
        dir_buttons_layout.addStretch()
        dir_layout.addLayout(dir_buttons_layout)
        
        layout.addWidget(dir_group)
        
        # Configuration group
        config_group = QGroupBox("Processing Configuration")
        config_layout = QVBoxLayout(config_group)
        
        # Configuration options
        config_options_layout = QHBoxLayout()
        
        # Left column
        left_column = QVBoxLayout()
        
        # Similarity threshold
        threshold_layout = QHBoxLayout()
        threshold_layout.addWidget(QLabel("Similarity Threshold:"))
        self.similarity_threshold = QDoubleSpinBox()
        self.similarity_threshold.setRange(0.1, 1.0)
        self.similarity_threshold.setValue(0.8)
        self.similarity_threshold.setSingleStep(0.05)
        self.similarity_threshold.setDecimals(2)
        threshold_layout.addWidget(self.similarity_threshold)
        threshold_layout.addStretch()
        left_column.addLayout(threshold_layout)
        
        # Batch size
        batch_layout = QHBoxLayout()
        batch_layout.addWidget(QLabel("Batch Size:"))
        self.batch_size = QSpinBox()
        self.batch_size.setRange(1, 64)  # Increased range for RTX 5080
        self.batch_size.setValue(16)  # Higher default for powerful GPU
        batch_layout.addWidget(self.batch_size)
        batch_layout.addStretch()
        left_column.addLayout(batch_layout)
        
        # Parallel processing
        self.parallel_processing_checkbox = QCheckBox("Enable Parallel Processing")
        self.parallel_processing_checkbox.setChecked(True)
        left_column.addWidget(self.parallel_processing_checkbox)
        
        # Number of jobs
        jobs_layout = QHBoxLayout()
        jobs_layout.addWidget(QLabel("CPU Jobs:"))
        self.n_jobs = QSpinBox()
        self.n_jobs.setRange(-1, 32)
        self.n_jobs.setValue(-1)  # Use all cores
        self.n_jobs.setToolTip("-1 = Use all available CPU cores")
        jobs_layout.addWidget(self.n_jobs)
        jobs_layout.addStretch()
        left_column.addLayout(jobs_layout)
        
        # Use GPU
        self.use_gpu_checkbox = QCheckBox("Use GPU (CUDA)")
        self.use_gpu_checkbox.setChecked(True)
        left_column.addWidget(self.use_gpu_checkbox)
        
        config_options_layout.addLayout(left_column)
        
        # Right column
        right_column = QVBoxLayout()
        
        # Quality metric
        quality_layout = QHBoxLayout()
        quality_layout.addWidget(QLabel("Quality Metric:"))
        self.quality_metric = QComboBox()
        self.quality_metric.addItems(["combined", "brisque", "niqe"])
        quality_layout.addWidget(self.quality_metric)
        quality_layout.addStretch()
        right_column.addLayout(quality_layout)
        
        # Save progress
        self.save_progress_checkbox = QCheckBox("Save Progress")
        self.save_progress_checkbox.setChecked(True)
        right_column.addWidget(self.save_progress_checkbox)
        
        right_column.addStretch()
        config_options_layout.addLayout(right_column)
        
        config_layout.addLayout(config_options_layout)
        layout.addWidget(config_group)
        
        # Progress group
        progress_group = QGroupBox("Processing Progress")
        progress_layout = QVBoxLayout(progress_group)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        progress_layout.addWidget(self.progress_bar)
        
        # Status text
        self.status_text = QTextEdit()
        self.status_text.setMaximumHeight(100)
        self.status_text.setReadOnly(True)
        progress_layout.addWidget(self.status_text)
        
        layout.addWidget(progress_group)
        
        # Control buttons
        control_layout = QHBoxLayout()
        
        self.start_button = QPushButton("Start Processing")
        self.start_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 4px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #45A049;
            }
            QPushButton:pressed {
                background-color: #3D8B40;
            }
            QPushButton:disabled {
                background-color: #666666;
            }
        """)
        self.start_button.clicked.connect(self.start_processing)
        control_layout.addWidget(self.start_button)
        
        self.stop_button = QPushButton("Stop Processing")
        self.stop_button.setStyleSheet("""
            QPushButton {
                background-color: #E94E77;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 4px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #D13A63;
            }
            QPushButton:pressed {
                background-color: #B82E4F;
            }
            QPushButton:disabled {
                background-color: #666666;
            }
        """)
        self.stop_button.clicked.connect(self.stop_processing)
        self.stop_button.setEnabled(False)
        control_layout.addWidget(self.stop_button)
        
        control_layout.addStretch()
        
        # Quick scan button
        self.quick_scan_button = QPushButton("Quick Scan")
        self.quick_scan_button.setStyleSheet("""
            QPushButton {
                background-color: #4A90E2;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 4px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #357ABD;
            }
            QPushButton:pressed {
                background-color: #2D5A8E;
            }
        """)
        self.quick_scan_button.clicked.connect(self.quick_scan)
        control_layout.addWidget(self.quick_scan_button)
        
        layout.addLayout(control_layout)
        layout.addStretch()
    
    def add_directory(self):
        """Add a directory to the scan list."""
        directory = QFileDialog.getExistingDirectory(
            self, "Select Directory to Scan"
        )
        
        if directory:
            path = Path(directory)
            if path not in self.source_directories:
                self.source_directories.append(path)
                self.dir_list.addItem(str(path))
                self.log_message(f"Added directory: {path}")
    
    def remove_directory(self):
        """Remove selected directory from the scan list."""
        current_row = self.dir_list.currentRow()
        if current_row >= 0:
            item = self.dir_list.takeItem(current_row)
            path = Path(item.text())
            self.source_directories.remove(path)
            self.log_message(f"Removed directory: {path}")
    
    def clear_directories(self):
        """Clear all directories from the scan list."""
        self.dir_list.clear()
        self.source_directories.clear()
        self.log_message("Cleared all directories")
    
    def start_scan(self):
        """Start the scanning process."""
        if not self.source_directories:
            self.add_directory()
            if not self.source_directories:
                return
        
        self.start_processing()
    
    def quick_scan(self):
        """Quick scan with default settings."""
        if not self.source_directories:
            self.add_directory()
            if not self.source_directories:
                return
        
        # Use default settings for quick scan
        self.similarity_threshold.setValue(0.8)
        self.batch_size.setValue(16)
        self.use_gpu_checkbox.setChecked(True)
        self.quality_metric.setCurrentText("combined")
        self.save_progress_checkbox.setChecked(True)
        
        self.start_processing()
    
    def start_processing(self):
        """Start the image processing."""
        if not self.source_directories:
            QMessageBox.warning(self, "No Directories", "Please add at least one directory to scan.")
            return
        
        # Create deduplication configuration
        config = DeduplicationConfig(
            similarity_threshold=self.similarity_threshold.value(),
            batch_size=self.batch_size.value(),
            use_gpu=self.use_gpu_checkbox.isChecked(),
            quality_metric=self.quality_metric.currentText(),
            save_progress=self.save_progress_checkbox.isChecked(),
            parallel_processing=self.parallel_processing_checkbox.isChecked(),
            n_jobs=self.n_jobs.value()
        )
        
        # Create deduplicator
        self.deduplicator = Deduplicator(self.db_manager, config)
        
        # Create and start processing thread
        self.processing_thread = ProcessingThread(self.deduplicator, self.source_directories)
        self.processing_thread.progress_updated.connect(self.update_progress)
        self.processing_thread.processing_finished.connect(self.on_processing_finished)
        self.processing_thread.error_occurred.connect(self.on_processing_error)
        
        # Update UI state
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.quick_scan_button.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        # Clear status text
        self.status_text.clear()
        self.log_message("Starting image processing...")
        
        # Start processing
        self.processing_thread.start()
        self.processing_started.emit()
    
    def stop_processing(self):
        """Stop the image processing."""
        if self.processing_thread and self.processing_thread.isRunning():
            self.processing_thread.stop()
            self.processing_thread.wait()
            self.log_message("Processing stopped by user")
        
        self.on_processing_finished()
    
    def update_progress(self, value: int, maximum: int, message: str):
        """Update progress display."""
        self.progress_bar.setMaximum(maximum)
        self.progress_bar.setValue(value)
        self.log_message(message)
        self.progress_updated.emit(value, maximum, message)
    
    def on_processing_finished(self, duplicate_groups=None):
        """Called when processing finishes."""
        # Update UI state
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.quick_scan_button.setEnabled(True)
        self.progress_bar.setVisible(False)
        
        if duplicate_groups:
            self.log_message(f"Processing completed! Found {len(duplicate_groups)} duplicate groups.")
            QMessageBox.information(
                self, "Processing Complete", 
                f"Processing completed successfully!\n\n"
                f"Found {len(duplicate_groups)} duplicate groups.\n"
                f"You can now review the duplicates in the Review tab."
            )
        else:
            self.log_message("Processing completed.")
        
        self.processing_finished.emit()
    
    def on_processing_error(self, error_message: str):
        """Called when processing encounters an error."""
        self.log_message(f"Error: {error_message}")
        QMessageBox.critical(self, "Processing Error", f"An error occurred during processing:\n\n{error_message}")
        self.on_processing_finished()
    
    def log_message(self, message: str):
        """Add a message to the status text."""
        self.status_text.append(f"[{QTimer().remainingTime()}] {message}")
        self.status_text.ensureCursorVisible() 