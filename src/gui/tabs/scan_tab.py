<<<<<<< HEAD
"""
Scan tab for Meme-Cleanup.

Handles directory scanning, image processing, and progress tracking.
"""

import logging
from pathlib import Path
from typing import List, Optional, Dict, Any

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QListWidget,
    QFileDialog, QProgressBar, QGroupBox, QCheckBox, QSpinBox, QDoubleSpinBox,
    QTextEdit, QMessageBox, QFrame, QComboBox, QProgressDialog, QDialog, QRadioButton,
    QButtonGroup
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QFont

from db.database import DatabaseManager
from utils.config_manager import ConfigManager
from core.deduplicator import Deduplicator, DeduplicationConfig
from utils.image_utils import get_scan_summary


logger = logging.getLogger(__name__)


class ProcessingThread(QThread):
    """Thread for running deduplication processing."""
    
    progress_updated = pyqtSignal(int, int, str)  # value, maximum, message
    processing_finished = pyqtSignal(list)  # duplicate groups
    error_occurred = pyqtSignal(str)  # error message
    scan_results_ready = pyqtSignal(dict)  # scan results
    
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
            self.scan_results_ready.emit(self.deduplicator.get_scan_results())
        except Exception as e:
            logger.error(f"Processing failed: {e}")
            self.error_occurred.emit(str(e))
    
    def stop(self):
        """Request stop of processing."""
        self._stop_requested = True


class ScanWorker(QThread):
    """Worker thread for scanning directories."""
    progress_updated = pyqtSignal(int, int)
    log_message = pyqtSignal(str)
    scan_completed = pyqtSignal(list)
    scan_failed = pyqtSignal(str)
    scan_results_ready = pyqtSignal(dict)


class ScanTab(QWidget):
    """Tab for scanning directories and processing images."""
    
    progress_updated = pyqtSignal(int, int, str)
    processing_started = pyqtSignal()
    processing_finished = pyqtSignal()
    
    def __init__(self, db_manager: DatabaseManager, config_manager: ConfigManager):
        """
        Initialize scan tab.
        
        Args:
            db_manager: Database manager instance
            config_manager: Configuration manager instance
        """
        super().__init__()
        self.db_manager = db_manager
        self.config_manager = config_manager
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
        
        # Directory summary
        self.dir_summary_label = QLabel("No directories added")
        self.dir_summary_label.setStyleSheet("color: #666; font-style: italic;")
        dir_layout.addWidget(self.dir_summary_label)
        
        # Directory buttons
        dir_buttons_layout = QHBoxLayout()
        
        self.add_dir_button = QPushButton("Add Directory")
        self.add_dir_button.clicked.connect(self.add_directory)
        dir_buttons_layout.addWidget(self.add_dir_button)
        
        self.add_multiple_button = QPushButton("Add Multiple")
        self.add_multiple_button.clicked.connect(self.quick_add_directories)
        self.add_multiple_button.setToolTip("Add multiple directories by entering paths separated by commas")
        dir_buttons_layout.addWidget(self.add_multiple_button)
        
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
        self.similarity_threshold.setValue(self.config_manager.processing.similarity_threshold)
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
        self.batch_size.setValue(self.config_manager.processing.batch_size)
        batch_layout.addWidget(self.batch_size)
        batch_layout.addStretch()
        left_column.addLayout(batch_layout)
        
        # Parallel processing
        self.parallel_processing_checkbox = QCheckBox("Enable Parallel Processing")
        self.parallel_processing_checkbox.setChecked(self.config_manager.processing.parallel_processing)
        left_column.addWidget(self.parallel_processing_checkbox)
        
        # Number of jobs
        jobs_layout = QHBoxLayout()
        jobs_layout.addWidget(QLabel("CPU Jobs:"))
        self.n_jobs = QSpinBox()
        self.n_jobs.setRange(-1, 32)
        self.n_jobs.setValue(self.config_manager.processing.n_jobs)
        self.n_jobs.setToolTip("-1 = Use all available CPU cores")
        jobs_layout.addWidget(self.n_jobs)
        jobs_layout.addStretch()
        left_column.addLayout(jobs_layout)
        
        # Use GPU
        self.use_gpu_checkbox = QCheckBox("Use GPU (CUDA)")
        self.use_gpu_checkbox.setChecked(self.config_manager.processing.use_gpu)
        left_column.addWidget(self.use_gpu_checkbox)
        
        config_options_layout.addLayout(left_column)
        
        # Right column
        right_column = QVBoxLayout()
        
        # Quality metric
        quality_layout = QHBoxLayout()
        quality_layout.addWidget(QLabel("Quality Metric:"))
        self.quality_metric = QComboBox()
        self.quality_metric.addItems(["combined", "brisque", "niqe"])
        self.quality_metric.setCurrentText(self.config_manager.processing.quality_metric)
        quality_layout.addWidget(self.quality_metric)
        quality_layout.addStretch()
        right_column.addLayout(quality_layout)
        
        # Save progress
        self.save_progress_checkbox = QCheckBox("Save Progress")
        self.save_progress_checkbox.setChecked(self.config_manager.processing.save_progress)
        right_column.addWidget(self.save_progress_checkbox)
        
        right_column.addStretch()
        config_options_layout.addLayout(right_column)
        
        config_layout.addLayout(config_options_layout)
        layout.addWidget(config_group)
        
        # Action buttons
        action_layout = QHBoxLayout()
        
        self.start_scan_button = QPushButton("Start Scan")
        self.start_scan_button.clicked.connect(self.start_scan)
        action_layout.addWidget(self.start_scan_button)
        
        self.start_processing_button = QPushButton("Start Processing")
        self.start_processing_button.clicked.connect(self.start_processing)
        self.start_processing_button.setEnabled(False)
        action_layout.addWidget(self.start_processing_button)
        
        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop_processing)
        self.stop_button.setEnabled(False)
        action_layout.addWidget(self.stop_button)
        
        action_layout.addStretch()
        
        # Test button
        self.test_button = QPushButton("Test Similarity")
        self.test_button.clicked.connect(self.test_similarity_detection)
        action_layout.addWidget(self.test_button)
        
        layout.addLayout(action_layout)
        
        # Progress and status
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # Log output
        self.log_output = QTextEdit()
        self.log_output.setMaximumHeight(150)
        self.log_output.setReadOnly(True)
        layout.addWidget(self.log_output)
        
        # Update directory summary
        self.update_directory_summary()
    
    def update_directory_summary(self):
        """Update the directory summary label."""
        if not self.source_directories:
            self.dir_summary_label.setText("No directories added")
        else:
            total_files = sum(1 for _ in self._get_all_files())
            self.dir_summary_label.setText(f"{len(self.source_directories)} directories, ~{total_files} files")
    
    def _get_all_files(self):
        """Get all files from source directories."""
        for directory in self.source_directories:
            if directory.exists():
                for file_path in directory.rglob('*'):
                    if file_path.is_file():
                        yield file_path
    
    def add_directory(self):
        """Add a single directory."""
        directory = QFileDialog.getExistingDirectory(
            self, "Select Directory", ""
        )
        
        if directory:
            dir_path = Path(directory)
            if dir_path not in self.source_directories:
                self.source_directories.append(dir_path)
                self.dir_list.addItem(str(dir_path))
                self.update_directory_summary()
                logger.info(f"Added directory: {dir_path}")
            else:
                QMessageBox.information(
                    self, "Directory Already Added",
                    "This directory is already in the list."
                )
    
    def add_multiple_directories(self, directories: List[str]):
        """Add multiple directories from a list."""
        for directory in directories:
            dir_path = Path(directory.strip())
            if dir_path.exists() and dir_path not in self.source_directories:
                self.source_directories.append(dir_path)
                self.dir_list.addItem(str(dir_path))
                logger.info(f"Added directory: {dir_path}")
        
        self.update_directory_summary()
    
    def quick_add_directories(self):
        """Quick add multiple directories via dialog."""
        from PyQt6.QtWidgets import QInputDialog
        
        directories, ok = QInputDialog.getMultiLineText(
            self, "Add Multiple Directories",
            "Enter directory paths (one per line):"
        )
        
        if ok and directories.strip():
            dir_list = [d.strip() for d in directories.split('\n') if d.strip()]
            self.add_multiple_directories(dir_list)
    
    def remove_directory(self):
        """Remove selected directory."""
        current_row = self.dir_list.currentRow()
        if current_row >= 0:
            removed_path = self.source_directories.pop(current_row)
            self.dir_list.takeItem(current_row)
            self.update_directory_summary()
            logger.info(f"Removed directory: {removed_path}")
    
    def clear_directories(self):
        """Clear all directories."""
        self.source_directories.clear()
        self.dir_list.clear()
        self.update_directory_summary()
        logger.info("Cleared all directories")
    
    def get_source_directories(self) -> List[str]:
        """Get list of source directory paths as strings."""
        return [str(d) for d in self.source_directories]
    
    def load_source_directories(self, directories: List[str]):
        """Load source directories from a list of paths."""
        self.clear_directories()
        self.add_multiple_directories(directories)
    
    def start_scan(self):
        """Start directory scanning."""
        if not self.source_directories:
            QMessageBox.warning(
                self, "No Directories",
                "Please add source directories first."
            )
            return
        
        # Update configuration from UI
        self._update_config_from_ui()
        
        # Create deduplicator with current config
        config = DeduplicationConfig(
            similarity_threshold=self.similarity_threshold.value(),
            batch_size=self.batch_size.value(),
            use_gpu=self.use_gpu_checkbox.isChecked(),
            quality_metric=self.quality_metric.currentText(),
            parallel_processing=self.parallel_processing_checkbox.isChecked(),
            n_jobs=self.n_jobs.value(),
            save_progress=self.save_progress_checkbox.isChecked()
        )
        
        self.deduplicator = Deduplicator(self.db_manager, config)
        
        # Start scanning
        try:
            self.log_message("Starting directory scan...")
            self.progress_bar.setVisible(True)
            self.progress_bar.setRange(0, 0)  # Indeterminate progress
            
            # Scan directories
            image_files = self.deduplicator.scan_directories(self.source_directories)
            
            self.log_message(f"Scan completed. Found {len(image_files)} images.")
            self.progress_bar.setVisible(False)
            
            # Enable processing button
            self.start_processing_button.setEnabled(True)
            
            # Show scan summary
            scan_results = self.deduplicator.get_scan_results()
            self.show_scan_summary(scan_results)
            
        except Exception as e:
            logger.error(f"Scan failed: {e}")
            self.log_message(f"Scan failed: {e}")
            self.progress_bar.setVisible(False)
            QMessageBox.critical(self, "Scan Error", f"Failed to scan directories: {e}")
    
    def quick_scan(self):
        """Quick scan - add directories if needed and start scan."""
        if not self.source_directories:
            self.add_directory()
            if not self.source_directories:
                return
        
        self.start_scan()
    
    def start_processing(self):
        """Start the full processing pipeline."""
        if not self.deduplicator:
            QMessageBox.warning(
                self, "No Scan Data",
                "Please run a scan first to collect image data."
            )
            return
        
        # Update configuration from UI
        self._update_config_from_ui()
        
        # Start processing thread
        self.processing_thread = ProcessingThread(self.deduplicator, self.source_directories)
        self.processing_thread.progress_updated.connect(self.update_progress)
        self.processing_thread.processing_finished.connect(self.on_processing_finished)
        self.processing_thread.error_occurred.connect(self.on_processing_error)
        self.processing_thread.scan_results_ready.connect(self.show_scan_summary)
        
        self.processing_thread.start()
        
        # Update UI state
        self.start_scan_button.setEnabled(False)
        self.start_processing_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        
        self.processing_started.emit()
        self.log_message("Processing started...")
    
    def _update_config_from_ui(self):
        """Update configuration manager with current UI values."""
        self.config_manager.update_processing_config(
            similarity_threshold=self.similarity_threshold.value(),
            batch_size=self.batch_size.value(),
            use_gpu=self.use_gpu_checkbox.isChecked(),
            quality_metric=self.quality_metric.currentText(),
            parallel_processing=self.parallel_processing_checkbox.isChecked(),
            n_jobs=self.n_jobs.value()
        )
    
    def stop_processing(self):
        """Stop the processing thread."""
        if self.processing_thread and self.processing_thread.isRunning():
            self.processing_thread.stop()
            self.processing_thread.wait()
        
        # Reset UI state
        self.start_scan_button.setEnabled(True)
        self.start_processing_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.progress_bar.setVisible(False)
        
        self.processing_finished.emit()
        self.log_message("Processing stopped.")
    
    def update_progress(self, value: int, maximum: int, message: str):
        """Update progress bar and emit signal."""
        self.progress_bar.setVisible(True)
        self.progress_bar.setMaximum(maximum)
        self.progress_bar.setValue(value)
        
        self.progress_updated.emit(value, maximum, message)
        self.log_message(message)
    
    def on_processing_finished(self, duplicate_groups=None):
        """Handle processing finished."""
        # Reset UI state
        self.start_scan_button.setEnabled(True)
        self.start_processing_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.progress_bar.setVisible(False)
        
        self.processing_finished.emit()
        
        if duplicate_groups:
            self.log_message(f"Processing completed. Found {len(duplicate_groups)} duplicate groups.")
            QMessageBox.information(
                self, "Processing Complete",
                f"Processing completed successfully!\n\n"
                f"Found {len(duplicate_groups)} duplicate groups.\n"
                f"Switch to the Review tab to examine duplicates."
            )
        else:
            self.log_message("Processing completed.")
    
    def on_processing_error(self, error_message: str):
        """Handle processing error."""
        self.log_message(f"Error: {error_message}")
        QMessageBox.critical(self, "Processing Error", f"Processing failed: {error_message}")
    
    def log_message(self, message: str):
        """Add message to log output."""
        self.log_output.append(f"[{QTimer().remainingTime()}] {message}")
    
    def show_scan_summary(self, scan_results: Dict[str, Any]):
        """Show scan results summary."""
        summary = get_scan_summary(scan_results)
        self.log_message("Scan Summary:")
        for line in summary.split('\n'):
            if line.strip():
                self.log_message(f"  {line}")
    
    def consolidate_files(self):
        """Consolidate files to output directory."""
        if not self.deduplicator:
            QMessageBox.warning(
                self, "No Data",
                "Please run processing first to generate duplicate data."
            )
            return
        
        # Get output directory
        output_dir = self.config_manager.paths.output_directory
        if not output_dir:
            output_dir = QFileDialog.getExistingDirectory(
                self, "Select Output Directory", ""
            )
            if not output_dir:
                return
            
            # Save to config
            self.config_manager.update_paths_config(output_directory=output_dir)
        
        output_path = Path(output_dir)
        
        try:
            # Show consolidation dialog
            dialog = ConsolidationDialog(self)
            if dialog.exec() == QDialog.DialogCode.Accepted:
                preserve_structure = dialog.preserve_structure_checkbox.isChecked()
                copy_mode = dialog.copy_mode_radio.isChecked()
                
                # Start consolidation
                self.log_message("Starting file consolidation...")
                self.progress_bar.setVisible(True)
                self.progress_bar.setRange(0, 0)
                
                results = self.deduplicator.consolidate_files(
                    output_path, 
                    preserve_structure=preserve_structure,
                    copy_mode=copy_mode
                )
                
                self.progress_bar.setVisible(False)
                
                # Show results
                self.log_message("Consolidation completed!")
                self.log_message(f"Files processed: {results.get('total_files', 0)}")
                self.log_message(f"Files copied: {results.get('copied_files', 0)}")
                self.log_message(f"Files moved: {results.get('moved_files', 0)}")
                self.log_message(f"Errors: {results.get('errors', 0)}")
                
                QMessageBox.information(
                    self, "Consolidation Complete",
                    f"File consolidation completed successfully!\n\n"
                    f"Files processed: {results.get('total_files', 0)}\n"
                    f"Files copied: {results.get('copied_files', 0)}\n"
                    f"Files moved: {results.get('moved_files', 0)}\n"
                    f"Errors: {results.get('errors', 0)}"
                )
                
        except Exception as e:
            logger.error(f"Consolidation failed: {e}")
            self.progress_bar.setVisible(False)
            self.log_message(f"Consolidation failed: {e}")
            QMessageBox.critical(self, "Consolidation Error", f"Failed to consolidate files: {e}")
    
    def test_similarity_detection(self):
        """Test similarity detection with sample images."""
        if not self.deduplicator:
            QMessageBox.warning(
                self, "No Data",
                "Please run a scan first to collect image data."
            )
            return
        
        try:
            # Test with different thresholds
            test_thresholds = [0.7, 0.8, 0.9]
            results = self.deduplicator.test_similarity_detection(test_thresholds)
            
            # Show results
            message = "Similarity Detection Test Results:\n\n"
            for threshold, count in results.items():
                message += f"Threshold {threshold}: {count} groups\n"
            
            QMessageBox.information(self, "Test Results", message)
            self.log_message("Similarity detection test completed.")
            
        except Exception as e:
            logger.error(f"Similarity test failed: {e}")
            QMessageBox.critical(self, "Test Error", f"Failed to run similarity test: {e}")
    
    def show_processing_stats(self):
        """Show processing statistics."""
        try:
            stats = self.deduplicator.get_file_summary()
            
            message = "Processing Statistics:\n\n"
            message += f"Total files: {stats.get('total_files', 0)}\n"
            message += f"Image files: {stats.get('image_files', 0)}\n"
            message += f"Video files: {stats.get('video_files', 0)}\n"
            message += f"Corrupted files: {stats.get('corrupted_files', 0)}\n"
            message += f"Other files: {stats.get('other_files', 0)}\n"
            message += f"Total size: {stats.get('total_size_mb', 0):.1f} MB"
            
            QMessageBox.information(self, "Statistics", message)
            
        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            QMessageBox.critical(self, "Error", f"Failed to get statistics: {e}")
    
    def clear_database(self):
        """Clear the database."""
        reply = QMessageBox.question(
            self, "Clear Database",
            "Are you sure you want to clear the database? This will remove all image data.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            try:
                self.db_manager.clear_database()
                self.deduplicator = None
                self.start_processing_button.setEnabled(False)
                self.log_message("Database cleared.")
                QMessageBox.information(self, "Database Cleared", "Database has been cleared successfully.")
                
            except Exception as e:
                logger.error(f"Failed to clear database: {e}")
                QMessageBox.critical(self, "Error", f"Failed to clear database: {e}")


class ConsolidationDialog(QDialog):
    """Dialog for consolidation options."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Consolidation Options")
        self.setModal(True)
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the dialog UI."""
        layout = QVBoxLayout(self)
        
        # Preserve structure option
        self.preserve_structure_checkbox = QCheckBox("Preserve directory structure")
        self.preserve_structure_checkbox.setChecked(True)
        self.preserve_structure_checkbox.setToolTip("Maintain original folder hierarchy in output")
        layout.addWidget(self.preserve_structure_checkbox)
        
        # Operation mode
        mode_group = QGroupBox("Operation Mode")
        mode_layout = QVBoxLayout(mode_group)
        
        self.copy_mode_radio = QRadioButton("Copy files (preserve originals)")
        self.copy_mode_radio.setChecked(True)
        mode_layout.addWidget(self.copy_mode_radio)
        
        self.move_mode_radio = QRadioButton("Move files (delete originals)")
        mode_layout.addWidget(self.move_mode_radio)
        
        layout.addWidget(mode_group)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.ok_button = QPushButton("OK")
        self.ok_button.clicked.connect(self.accept)
        button_layout.addWidget(self.ok_button)
        
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(self.cancel_button)
        
        layout.addLayout(button_layout) 
=======
"""
Scan tab for Meme-Cleanup.

Handles directory scanning, image processing, and progress tracking.
"""

import logging
from pathlib import Path
from typing import List, Optional, Dict, Any

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QListWidget,
    QFileDialog, QProgressBar, QGroupBox, QCheckBox, QSpinBox, QDoubleSpinBox,
    QTextEdit, QMessageBox, QFrame, QComboBox, QProgressDialog, QDialog, QRadioButton,
    QButtonGroup
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QFont

from db.database import DatabaseManager
from core.deduplicator import Deduplicator, DeduplicationConfig
from utils.image_utils import get_scan_summary


logger = logging.getLogger(__name__)


class ProcessingThread(QThread):
    """Thread for running deduplication processing."""
    
    progress_updated = pyqtSignal(int, int, str)  # value, maximum, message
    processing_finished = pyqtSignal(list)  # duplicate groups
    error_occurred = pyqtSignal(str)  # error message
    scan_results_ready = pyqtSignal(dict)  # scan results
    
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
            self.scan_results_ready.emit(self.deduplicator.get_scan_results())
        except Exception as e:
            logger.error(f"Processing failed: {e}")
            self.error_occurred.emit(str(e))
    
    def stop(self):
        """Request stop of processing."""
        self._stop_requested = True


class ScanWorker(QThread):
    """Worker thread for scanning directories."""
    progress_updated = pyqtSignal(int, int)
    log_message = pyqtSignal(str)
    scan_completed = pyqtSignal(list)
    scan_failed = pyqtSignal(str)
    scan_results_ready = pyqtSignal(dict)


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
        
        # Directory summary
        self.dir_summary_label = QLabel("No directories added")
        self.dir_summary_label.setStyleSheet("color: #666; font-style: italic;")
        dir_layout.addWidget(self.dir_summary_label)
        
        # Directory buttons
        dir_buttons_layout = QHBoxLayout()
        
        self.add_dir_button = QPushButton("Add Directory")
        self.add_dir_button.clicked.connect(self.add_directory)
        dir_buttons_layout.addWidget(self.add_dir_button)
        
        self.add_multiple_button = QPushButton("Add Multiple")
        self.add_multiple_button.clicked.connect(self.quick_add_directories)
        self.add_multiple_button.setToolTip("Add multiple directories by entering paths separated by commas")
        dir_buttons_layout.addWidget(self.add_multiple_button)
        
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
        
        self.quick_scan_button = QPushButton("Quick Scan")
        self.quick_scan_button.setStyleSheet("""
            QPushButton {
                background-color: #FF9800;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 4px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #F57C00;
            }
            QPushButton:pressed {
                background-color: #E65100;
            }
            QPushButton:disabled {
                background-color: #666666;
            }
        """)
        self.quick_scan_button.clicked.connect(self.quick_scan)
        control_layout.addWidget(self.quick_scan_button)
        
        self.consolidate_button = QPushButton("Consolidate Files")
        self.consolidate_button.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 4px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:pressed {
                background-color: #0D47A1;
            }
            QPushButton:disabled {
                background-color: #666666;
            }
        """)
        self.consolidate_button.clicked.connect(self.consolidate_files)
        self.consolidate_button.setEnabled(False)
        control_layout.addWidget(self.consolidate_button)
        
        self.test_similarity_button = QPushButton("Test Similarity")
        self.test_similarity_button.setStyleSheet("""
            QPushButton {
                background-color: #9C27B0;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 4px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #7B1FA2;
            }
            QPushButton:pressed {
                background-color: #4A148C;
            }
            QPushButton:disabled {
                background-color: #666666;
            }
        """)
        self.test_similarity_button.clicked.connect(self.test_similarity_detection)
        self.test_similarity_button.setEnabled(False)
        control_layout.addWidget(self.test_similarity_button)
        
        self.show_stats_button = QPushButton("Show Stats")
        self.show_stats_button.setStyleSheet("""
            QPushButton {
                background-color: #607D8B;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 4px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #455A64;
            }
            QPushButton:pressed {
                background-color: #263238;
            }
            QPushButton:disabled {
                background-color: #666666;
            }
        """)
        self.show_stats_button.clicked.connect(self.show_processing_stats)
        self.show_stats_button.setEnabled(False)
        control_layout.addWidget(self.show_stats_button)
        
        self.clear_db_button = QPushButton("Clear DB")
        self.clear_db_button.setStyleSheet("""
            QPushButton {
                background-color: #F44336;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 4px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #D32F2F;
            }
            QPushButton:pressed {
                background-color: #B71C1C;
            }
            QPushButton:disabled {
                background-color: #666666;
            }
        """)
        self.clear_db_button.clicked.connect(self.clear_database)
        self.clear_db_button.setToolTip("Clear all data from database and start fresh")
        control_layout.addWidget(self.clear_db_button)
        
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
        
        layout.addLayout(control_layout)
        layout.addStretch()
    
    def update_directory_summary(self):
        """Update the directory summary display."""
        if not self.source_directories:
            self.dir_summary_label.setText("No directories added")
            return
        
        total_dirs = len(self.source_directories)
        self.dir_summary_label.setText(f"{total_dirs} directory{'s' if total_dirs > 1 else ''} added")
    
    def add_directory(self):
        """Add directories to the scan list."""
        # Use QFileDialog to select multiple directories
        directories = QFileDialog.getExistingDirectory(
            self, "Select Directories to Scan", "",
            QFileDialog.Option.ShowDirsOnly
        )
        
        if directories:
            path = Path(directories)
            if path not in self.source_directories:
                # Quick check - just verify it's a valid directory
                if not path.exists() or not path.is_dir():
                    QMessageBox.warning(self, "Invalid Directory", f"'{path}' is not a valid directory.")
                    return
                
                # Quick count of files
                try:
                    all_files = list(path.rglob('*'))
                    image_files = [f for f in all_files if f.is_file() and 
                                 f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp', '.gif'}]
                    video_files = [f for f in all_files if f.is_file() and 
                                 f.suffix.lower() in {'.mp4', '.mkv', '.avi', '.mov', '.wmv', '.flv', '.webm', '.m4v'}]
                    
                    count_text = f"""
Directory: {path.name}
Total files: {len(all_files)}
Image files: {len(image_files)}
Video files: {len(video_files)}
Other files: {len(all_files) - len(image_files) - len(video_files)}
                    """
                    
                    QMessageBox.information(self, "Directory Summary", count_text)
                    
                except Exception as e:
                    logger.warning(f"Could not count files in {path}: {e}")
                
                # Add to list without scanning
                self.source_directories.append(path)
                self.dir_list.addItem(str(path))
                self.log_message(f"Added directory: {path}")
                self.update_directory_summary()
    
    def add_multiple_directories(self):
        """Add multiple directories at once."""
        # Note: QFileDialog doesn't support multiple directory selection natively
        # So we'll use a custom dialog or just allow multiple single selections
        directories = QFileDialog.getExistingDirectory(
            self, "Select Directory to Add", "",
            QFileDialog.Option.ShowDirsOnly
        )
        
        if directories:
            path = Path(directories)
            if path not in self.source_directories:
                if not path.exists() or not path.is_dir():
                    QMessageBox.warning(self, "Invalid Directory", f"'{path}' is not a valid directory.")
                    return
                
                self.source_directories.append(path)
                self.dir_list.addItem(str(path))
                self.log_message(f"Added directory: {path}")
                self.update_directory_summary()
    
    def quick_add_directories(self):
        """Quick add multiple directories with a simple dialog."""
        from PyQt6.QtWidgets import QInputDialog
        
        # Get a comma-separated list of directories
        text, ok = QInputDialog.getText(
            self, "Add Multiple Directories", 
            "Enter directory paths (separated by commas):\n"
            "Example: E:\\Phone, E:\\Phone2, E:\\Phone3"
        )
        
        if ok and text.strip():
            directories = [d.strip() for d in text.split(',') if d.strip()]
            added_count = 0
            
            for dir_path in directories:
                path = Path(dir_path)
                if path not in self.source_directories:
                    if not path.exists() or not path.is_dir():
                        self.log_message(f"Skipped invalid directory: {path}")
                        continue
                    
                    self.source_directories.append(path)
                    self.dir_list.addItem(str(path))
                    self.log_message(f"Added directory: {path}")
                    added_count += 1
            
            if added_count > 0:
                self.update_directory_summary()
                self.log_message(f"Added {added_count} directories")
    
    def remove_directory(self):
        """Remove selected directory from the scan list."""
        current_row = self.dir_list.currentRow()
        if current_row >= 0:
            item = self.dir_list.takeItem(current_row)
            path = Path(item.text())
            self.source_directories.remove(path)
            self.log_message(f"Removed directory: {path}")
            self.update_directory_summary()
    
    def clear_directories(self):
        """Clear all directories from the scan list."""
        self.dir_list.clear()
        self.source_directories.clear()
        self.log_message("Cleared all directories")
        self.update_directory_summary()
    
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
        
        # Show processing summary
        summary_text = f"""
Processing Summary:

Directories to scan: {len(self.source_directories)}
Directories:
"""
        for i, directory in enumerate(self.source_directories, 1):
            summary_text += f"  {i}. {directory}\n"
        
        summary_text += f"""
This will:
1. Scan all directories for image files
2. Validate and process each image
3. Generate CLIP embeddings for similarity detection
4. Calculate quality metrics (BRISQUE/NIQE)
5. Find and group duplicate images
6. Select the best quality image from each group

Processing may take several minutes depending on the number of images.
        """
        
        reply = QMessageBox.question(
            self, "Start Processing", 
            summary_text,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes
        )
        
        if reply != QMessageBox.StandardButton.Yes:
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
        self.processing_thread.scan_results_ready.connect(self.show_scan_summary)
        
        # Update UI state
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.quick_scan_button.setEnabled(False)
        self.add_dir_button.setEnabled(False)
        self.add_multiple_button.setEnabled(False)
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
        """Update progress bar and status."""
        self.progress_bar.setValue(value)
        self.progress_bar.setMaximum(maximum)
        
        if message:
            self.log_message(message)
            
        # Update progress percentage
        if maximum > 0:
            percentage = (value / maximum) * 100
            self.progress_bar.setFormat(f"{percentage:.1f}% - {message}")
        else:
            self.progress_bar.setFormat(message)
    
    def on_processing_finished(self, duplicate_groups=None):
        """Handle processing completion."""
        # Update UI state
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.quick_scan_button.setEnabled(True)
        self.add_dir_button.setEnabled(True)
        self.add_multiple_button.setEnabled(True)
        self.progress_bar.setVisible(False)
        
        # Enable consolidate button
        self.consolidate_button.setEnabled(True)
        self.test_similarity_button.setEnabled(True)
        self.show_stats_button.setEnabled(True)
        
        if duplicate_groups:
            self.log_message(f"Processing completed. Found {len(duplicate_groups)} duplicate groups.")
        else:
            self.log_message("Processing completed. No duplicates found.")
            self.log_message("Try clicking 'Test Similarity' to see if lower thresholds find duplicates.")
        
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
    
    def show_scan_summary(self, scan_results: Dict[str, Any]):
        """Show a summary of the scan results."""
        summary_text = f"""
Scan Completed Successfully!

Directories Scanned: {scan_results.get('directories_scanned', 0)}
Total Images Found: {scan_results.get('total_images_found', 0)}
Valid Images Processed: {scan_results.get('total_images_processed', 0)}
Corrupted Files Skipped: {scan_results.get('corrupted_files_skipped', 0)}
Scan Duration: {scan_results.get('scan_duration', 0):.2f} seconds

The scan has completed and all valid images have been processed.
Corrupted files have been automatically skipped and will not be included in deduplication.
        """
        
        QMessageBox.information(self, "Scan Complete", summary_text)
    
    def consolidate_files(self):
        """Consolidate files to a single output directory."""
        if not self.deduplicator:
            QMessageBox.warning(self, "No Processing", "Please run processing first to identify duplicates.")
            return
        
        # Get file summary first
        try:
            file_summary = self.deduplicator.get_file_summary()
            
            summary_text = f"""
File Summary:

Total Files Found: {file_summary['total_files']}
Processed Images: {file_summary['processed_images_count']}
Video Files: {file_summary['video_files_count']}
Corrupted Files: {file_summary['corrupted_files_count']}
Unprocessed Files: {file_summary['unprocessed_files_count']}

What would you like to consolidate?
            """
            
            # Show summary and get user choice
            reply = QMessageBox.question(
                self, "File Summary", 
                summary_text,
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes
            )
            
            if reply != QMessageBox.StandardButton.Yes:
                return
                
        except Exception as e:
            logger.error(f"Failed to get file summary: {e}")
            # Continue with basic consolidation if summary fails
        
        # Get output directory
        output_directory = QFileDialog.getExistingDirectory(
            self, "Select Output Directory for Consolidated Files"
        )
        
        if not output_directory:
            return
        
        output_path = Path(output_directory)
        
        # Ask user for consolidation options
        dialog = QDialog(self)
        dialog.setWindowTitle("Consolidation Options")
        dialog.setModal(True)
        dialog.setMinimumWidth(400)
        
        layout = QVBoxLayout(dialog)
        
        # Structure preservation
        preserve_structure_checkbox = QCheckBox("Preserve subdirectory structure")
        preserve_structure_checkbox.setChecked(True)
        preserve_structure_checkbox.setToolTip("Keep the original folder structure from source directories")
        layout.addWidget(preserve_structure_checkbox)
        
        # Copy vs Move
        copy_radio = QRadioButton("Copy files (keep originals)")
        copy_radio.setChecked(True)
        move_radio = QRadioButton("Move files (delete originals)")
        
        copy_group = QButtonGroup()
        copy_group.addButton(copy_radio)
        copy_group.addButton(move_radio)
        
        layout.addWidget(copy_radio)
        layout.addWidget(move_radio)
        
        # File type options
        layout.addWidget(QLabel("\nFile Types to Include:"))
        
        include_videos_checkbox = QCheckBox("Include video files")
        include_videos_checkbox.setChecked(True)
        include_videos_checkbox.setToolTip("Copy/move video files (.mp4, .mkv, etc.)")
        layout.addWidget(include_videos_checkbox)
        
        include_corrupted_checkbox = QCheckBox("Include corrupted files")
        include_corrupted_checkbox.setChecked(False)
        include_corrupted_checkbox.setToolTip("Copy/move files that failed validation (for manual review)")
        layout.addWidget(include_corrupted_checkbox)
        
        include_unprocessed_checkbox = QCheckBox("Include unprocessed images")
        include_unprocessed_checkbox.setChecked(True)
        include_unprocessed_checkbox.setToolTip("Copy/move valid images that weren't processed (for manual review)")
        layout.addWidget(include_unprocessed_checkbox)
        
        # Buttons
        button_layout = QHBoxLayout()
        ok_button = QPushButton("Start Consolidation")
        cancel_button = QPushButton("Cancel")
        
        ok_button.clicked.connect(dialog.accept)
        cancel_button.clicked.connect(dialog.reject)
        
        button_layout.addWidget(ok_button)
        button_layout.addWidget(cancel_button)
        layout.addLayout(button_layout)
        
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        
        # Start consolidation
        try:
            preserve_structure = preserve_structure_checkbox.isChecked()
            copy_mode = copy_radio.isChecked()
            include_videos = include_videos_checkbox.isChecked()
            include_corrupted = include_corrupted_checkbox.isChecked()
            include_unprocessed = include_unprocessed_checkbox.isChecked()
            
            self.log_message(f"Starting comprehensive consolidation to {output_path}")
            self.log_message(f"Mode: {'Copy' if copy_mode else 'Move'}, "
                           f"Preserve structure: {preserve_structure}")
            self.log_message(f"Include videos: {include_videos}, "
                           f"Corrupted: {include_corrupted}, "
                           f"Unprocessed: {include_unprocessed}")
            
            # Run comprehensive consolidation
            results = self.deduplicator.consolidate_all_files(
                output_path, 
                preserve_structure=preserve_structure,
                copy_mode=copy_mode,
                include_videos=include_videos,
                include_corrupted=include_corrupted
            )
            
            # Show results
            result_text = f"""
Comprehensive Consolidation Completed!

Output Directory: {output_path}

Results:
"""
            
            for file_type, stats in results.items():
                if file_type != 'errors':
                    result_text += f"  {file_type.replace('_', ' ').title()}: {stats['successful']} successful, {stats['failed']} failed\n"
            
            if results['errors']:
                result_text += f"\nErrors:\n"
                for error in results['errors'][:5]:  # Show first 5 errors
                    result_text += f"• {error}\n"
                if len(results['errors']) > 5:
                    result_text += f"... and {len(results['errors']) - 5} more errors"
            
            QMessageBox.information(self, "Consolidation Complete", result_text)
            
            total_successful = sum(stats['successful'] for file_type, stats in results.items() if file_type != 'errors')
            self.log_message(f"Comprehensive consolidation completed: {total_successful} files processed")
            
        except Exception as e:
            logger.error(f"Consolidation failed: {e}")
            QMessageBox.critical(self, "Consolidation Failed", f"Failed to consolidate files: {e}")
            self.log_message(f"Consolidation failed: {e}")

    def test_similarity_detection(self):
        """Test similarity detection with different thresholds."""
        if not self.deduplicator:
            QMessageBox.warning(self, "No Processing", "Please run processing first.")
            return
        
        try:
            self.log_message("Testing similarity detection with different thresholds...")
            
            # Test with different thresholds
            results = self.deduplicator.test_similarity_detection()
            
            if not results:
                QMessageBox.information(self, "Test Results", "No processed images found to test.")
                return
            
            # Show results
            result_text = "Similarity Detection Test Results:\n\n"
            
            for threshold_key, threshold_results in results.items():
                threshold = threshold_key.replace('threshold_', '')
                result_text += f"Threshold {threshold}:\n"
                result_text += f"  Groups found: {threshold_results['groups_found']}\n"
                result_text += f"  Total images in groups: {threshold_results['total_images_in_groups']}\n"
                
                if threshold_results['groups']:
                    result_text += f"  Sample groups:\n"
                    for i, group in enumerate(threshold_results['groups'][:3]):  # Show first 3 groups
                        result_text += f"    Group {i+1}:\n"
                        for img in group['images']:
                            result_text += f"      • {Path(img['file_path']).name} (score: {img['similarity_score']:.3f})\n"
                
                result_text += "\n"
            
            # Add recommendation
            best_threshold = None
            best_groups = 0
            for threshold_key, threshold_results in results.items():
                if threshold_results['groups_found'] > best_groups:
                    best_groups = threshold_results['groups_found']
                    best_threshold = threshold_key.replace('threshold_', '')
            
            if best_threshold:
                result_text += f"Recommendation: Try threshold {best_threshold} for better duplicate detection.\n"
                result_text += f"Current threshold: {self.similarity_threshold.value()}"
            
            QMessageBox.information(self, "Similarity Test Results", result_text)
            self.log_message(f"Similarity test completed. Best threshold: {best_threshold}")
            
        except Exception as e:
            logger.error(f"Similarity test failed: {e}")
            QMessageBox.critical(self, "Test Failed", f"Failed to test similarity detection: {e}")
            self.log_message(f"Similarity test failed: {e}")

    def show_processing_stats(self):
        """Show detailed processing statistics."""
        if not self.deduplicator:
            QMessageBox.warning(self, "No Processing", "Please run processing first.")
            return
        
        try:
            # Get scan results
            scan_results = self.deduplicator.get_scan_results()
            
            if not scan_results:
                QMessageBox.information(self, "No Data", "No scan results available.")
                return
            
            # Get file summary
            file_summary = self.deduplicator.get_file_summary()
            
            stats_text = f"""
Processing Statistics:

Scan Results:
  Directories scanned: {scan_results.get('directories_scanned', 0)}
  Total images found: {scan_results.get('total_images_found', 0)}
  Successfully processed: {scan_results.get('total_images_processed', 0)}
  Corrupted files skipped: {scan_results.get('corrupted_files_skipped', 0)}
  Other files skipped: {scan_results.get('skipped_files_count', 0)}
  Scan duration: {scan_results.get('scan_duration', 0):.2f} seconds

Current File Status:
  Processed images: {file_summary.get('processed_images_count', 0)}
  Video files: {file_summary.get('video_files_count', 0)}
  Corrupted files: {file_summary.get('corrupted_files_count', 0)}
  Unprocessed files: {file_summary.get('unprocessed_files_count', 0)}
  Total files: {file_summary.get('total_files', 0)}

Processing Rate:
  Images per second: {scan_results.get('total_images_processed', 0) / max(scan_results.get('scan_duration', 1), 1):.1f}
            """
            
            QMessageBox.information(self, "Processing Statistics", stats_text)
            
        except Exception as e:
            logger.error(f"Failed to get processing stats: {e}")
            QMessageBox.critical(self, "Error", f"Failed to get processing statistics: {e}")

    def clear_database(self):
        """Clear all data from the database."""
        reply = QMessageBox.question(
            self, "Clear Database", 
            "Are you sure you want to clear all data from the database? This action cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            self.db_manager.clear_database()
            self.log_message("Database cleared")
            self.on_processing_finished()
        else:
            self.log_message("Database clearing cancelled") 
>>>>>>> 88a757b8ecc4ae4d819ae02a34c56b2a1ebc3714
