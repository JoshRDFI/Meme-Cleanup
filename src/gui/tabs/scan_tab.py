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
        self.quality_metric.addItems(["BRISQUE", "NIQE", "Combined"])
        self.quality_metric.setCurrentText(self.config_manager.processing.quality_metric)
        quality_layout.addWidget(self.quality_metric)
        quality_layout.addStretch()
        right_column.addLayout(quality_layout)
        
        # Skip corrupted files
        self.skip_corrupted_checkbox = QCheckBox("Skip Corrupted Files")
        self.skip_corrupted_checkbox.setChecked(self.config_manager.processing.skip_corrupted)
        right_column.addWidget(self.skip_corrupted_checkbox)
        
        # Skip animated files
        self.skip_animated_checkbox = QCheckBox("Skip Animated Files (GIF, WebP)")
        self.skip_animated_checkbox.setChecked(self.config_manager.processing.skip_animated)
        right_column.addWidget(self.skip_animated_checkbox)
        
        # Skip small files
        skip_small_layout = QHBoxLayout()
        skip_small_layout.addWidget(QLabel("Skip files smaller than:"))
        self.min_file_size = QSpinBox()
        self.min_file_size.setRange(0, 1000000)
        self.min_file_size.setValue(self.config_manager.processing.min_file_size)
        self.min_file_size.setSuffix(" bytes")
        skip_small_layout.addWidget(self.min_file_size)
        skip_small_layout.addStretch()
        right_column.addLayout(skip_small_layout)
        
        # Save progress
        self.save_progress_checkbox = QCheckBox("Save Progress Automatically")
        self.save_progress_checkbox.setChecked(self.config_manager.processing.save_progress)
        right_column.addWidget(self.save_progress_checkbox)
        
        config_options_layout.addLayout(right_column)
        config_layout.addLayout(config_options_layout)
        
        layout.addWidget(config_group)
        
        # Progress and controls
        progress_group = QGroupBox("Processing")
        progress_layout = QVBoxLayout(progress_group)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        progress_layout.addWidget(self.progress_bar)
        
        # Status label
        self.status_label = QLabel("Ready to scan")
        self.status_label.setStyleSheet("color: #666; font-style: italic;")
        progress_layout.addWidget(self.status_label)
        
        # Control buttons
        buttons_layout = QHBoxLayout()
        
        self.start_button = QPushButton("Start Processing")
        self.start_button.setStyleSheet("""
            QPushButton {
                background-color: #4A90E2;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #357ABD;
            }
            QPushButton:pressed {
                background-color: #2D5A8E;
            }
        """)
        self.start_button.clicked.connect(self.start_processing)
        buttons_layout.addWidget(self.start_button)
        
        self.stop_button = QPushButton("Stop Processing")
        self.stop_button.setStyleSheet("""
            QPushButton {
                background-color: #E94E77;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #D13A63;
            }
            QPushButton:pressed {
                background-color: #B82E4F;
            }
        """)
        self.stop_button.clicked.connect(self.stop_processing)
        self.stop_button.setEnabled(False)
        buttons_layout.addWidget(self.stop_button)
        
        buttons_layout.addStretch()
        
        # Additional buttons
        self.consolidate_button = QPushButton("Consolidate Files")
        self.consolidate_button.clicked.connect(self.consolidate_files)
        self.consolidate_button.setEnabled(False)
        buttons_layout.addWidget(self.consolidate_button)
        
        self.stats_button = QPushButton("Show Statistics")
        self.stats_button.clicked.connect(self.show_processing_stats)
        buttons_layout.addWidget(self.stats_button)
        
        progress_layout.addLayout(buttons_layout)
        layout.addWidget(progress_group)
        
        # Log output
        log_group = QGroupBox("Processing Log")
        log_layout = QVBoxLayout(log_group)
        
        self.log_text = QTextEdit()
        self.log_text.setMaximumHeight(200)
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)
        
        layout.addWidget(log_group)
        
        # Load saved directories from config
        self.load_saved_directories()
    
    def load_saved_directories(self):
        """Load saved directories from configuration."""
        saved_dirs = self.config_manager.paths.source_directories
        if saved_dirs:
            for dir_path in saved_dirs:
                if Path(dir_path).exists():
                    self.add_directory_to_list(dir_path)
            self.update_directory_summary()
    
    def update_directory_summary(self):
        """Update the directory summary label."""
        if not self.source_directories:
            self.dir_summary_label.setText("No directories added")
            return
        
        total_files = sum(len(list(Path(d).rglob("*"))) for d in self.source_directories)
        self.dir_summary_label.setText(
            f"{len(self.source_directories)} directories, ~{total_files} files"
        )
    
    def _get_all_files(self):
        """Get all files from source directories."""
        all_files = []
        for directory in self.source_directories:
            all_files.extend(Path(directory).rglob("*"))
        return all_files
    
    def add_directory(self):
        """Add a single directory."""
        directory = QFileDialog.getExistingDirectory(
            self, "Select Directory", str(Path.home())
        )
        if directory:
            self.add_directory_to_list(directory)
            self.update_directory_summary()
    
    def add_directory_to_list(self, directory: str):
        """Add directory to the list and source directories."""
        if directory not in self.source_directories:
            self.source_directories.append(directory)
            self.dir_list.addItem(directory)
    
    def add_multiple_directories(self, directories: List[str]):
        """Add multiple directories from a list."""
        for directory in directories:
            if Path(directory).exists() and directory not in self.source_directories:
                self.add_directory_to_list(directory)
        self.update_directory_summary()
    
    def quick_add_directories(self):
        """Quick add multiple directories via dialog."""
        text, ok = QInputDialog.getText(
            self, "Add Multiple Directories",
            "Enter directory paths (separated by commas):"
        )
        if ok and text:
            directories = [d.strip() for d in text.split(",") if d.strip()]
            self.add_multiple_directories(directories)
    
    def remove_directory(self):
        """Remove selected directory."""
        current_row = self.dir_list.currentRow()
        if current_row >= 0:
            directory = self.source_directories.pop(current_row)
            self.dir_list.takeItem(current_row)
            self.update_directory_summary()
    
    def clear_directories(self):
        """Clear all directories."""
        self.source_directories.clear()
        self.dir_list.clear()
        self.update_directory_summary()
    
    def get_source_directories(self) -> List[str]:
        """Get list of source directories."""
        return self.source_directories.copy()
    
    def load_source_directories(self, directories: List[str]):
        """Load source directories from a list."""
        self.clear_directories()
        self.add_multiple_directories(directories)
    
    def start_scan(self):
        """Start the scanning and processing."""
        if not self.source_directories:
            QMessageBox.warning(
                self, "No Directories",
                "Please add at least one source directory first."
            )
            return
        
        # Update configuration from UI
        self._update_config_from_ui()
        
        # Save directories to config
        self.config_manager.paths.source_directories = self.source_directories
        self.config_manager.save_config()
        
        # Start processing
        self.start_processing()
    
    def quick_scan(self):
        """Quick scan with default settings."""
        if not self.source_directories:
            self.add_directory()
        if self.source_directories:
            self.start_scan()
    
    def start_processing(self):
        """Start the deduplication processing."""
        if not self.source_directories:
            QMessageBox.warning(
                self, "No Directories",
                "Please add at least one source directory first."
            )
            return
        
        try:
            # Update configuration from UI
            self._update_config_from_ui()
            
            # Create deduplicator with current configuration
            config = DeduplicationConfig(
                similarity_threshold=self.similarity_threshold.value(),
                batch_size=self.batch_size.value(),
                parallel_processing=self.parallel_processing_checkbox.isChecked(),
                n_jobs=self.n_jobs.value(),
                use_gpu=self.use_gpu_checkbox.isChecked(),
                quality_metric=self.quality_metric.currentText(),
                skip_corrupted=self.skip_corrupted_checkbox.isChecked(),
                skip_animated=self.skip_animated_checkbox.isChecked(),
                min_file_size=self.min_file_size.value(),
                save_progress=self.save_progress_checkbox.isChecked()
            )
            
            self.deduplicator = Deduplicator(self.db_manager, config)
            
            # Create and start processing thread
            self.processing_thread = ProcessingThread(
                self.deduplicator, 
                [Path(d) for d in self.source_directories]
            )
            self.processing_thread.progress_updated.connect(self.update_progress)
            self.processing_thread.processing_finished.connect(self.on_processing_finished)
            self.processing_thread.error_occurred.connect(self.on_processing_error)
            self.processing_thread.scan_results_ready.connect(self.show_scan_summary)
            
            # Update UI state
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            self.progress_bar.setVisible(True)
            self.status_label.setText("Processing started...")
            
            # Emit signals
            self.processing_started.emit()
            
            # Start processing
            self.processing_thread.start()
            
            logger.info("Processing started")
            
        except Exception as e:
            logger.error(f"Failed to start processing: {e}")
            QMessageBox.critical(
                self, "Error",
                f"Failed to start processing: {e}"
            )
    
    def _update_config_from_ui(self):
        """Update configuration manager with current UI values."""
        self.config_manager.processing.similarity_threshold = self.similarity_threshold.value()
        self.config_manager.processing.batch_size = self.batch_size.value()
        self.config_manager.processing.parallel_processing = self.parallel_processing_checkbox.isChecked()
        self.config_manager.processing.n_jobs = self.n_jobs.value()
        self.config_manager.processing.use_gpu = self.use_gpu_checkbox.isChecked()
        self.config_manager.processing.quality_metric = self.quality_metric.currentText()
        self.config_manager.processing.skip_corrupted = self.skip_corrupted_checkbox.isChecked()
        self.config_manager.processing.skip_animated = self.skip_animated_checkbox.isChecked()
        self.config_manager.processing.min_file_size = self.min_file_size.value()
        self.config_manager.processing.save_progress = self.save_progress_checkbox.isChecked()
        self.config_manager.save_config()
    
    def stop_processing(self):
        """Stop the processing."""
        if self.processing_thread and self.processing_thread.isRunning():
            self.processing_thread.stop()
            self.processing_thread.wait()
        
        # Update UI state
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.progress_bar.setVisible(False)
        self.status_label.setText("Processing stopped")
        
        logger.info("Processing stopped")
    
    def update_progress(self, value: int, maximum: int, message: str):
        """Update progress bar and status."""
        self.progress_bar.setMaximum(maximum)
        self.progress_bar.setValue(value)
        self.status_label.setText(message)
        
        # Emit signal for main window
        self.progress_updated.emit(value, maximum, message)
    
    def on_processing_finished(self, duplicate_groups=None):
        """Handle processing finished."""
        # Update UI state
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.progress_bar.setVisible(False)
        self.status_label.setText("Processing completed")
        
        # Enable consolidate button if we have duplicates
        if duplicate_groups:
            self.consolidate_button.setEnabled(True)
        
        # Log completion
        self.log_message("Processing completed successfully")
        
        # Emit signal for main window
        self.processing_finished.emit()
        
        logger.info("Processing completed")
    
    def on_processing_error(self, error_message: str):
        """Handle processing error."""
        self.log_message(f"Error: {error_message}")
        self.status_label.setText("Processing failed")
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.progress_bar.setVisible(False)
        
        QMessageBox.critical(
            self, "Processing Error",
            f"An error occurred during processing:\n\n{error_message}"
        )
    
    def log_message(self, message: str):
        """Add message to log."""
        self.log_text.append(f"[{QTimer().remainingTime()}] {message}")
    
    def show_scan_summary(self, scan_results: Dict[str, Any]):
        """Show scan results summary."""
        summary = get_scan_summary(scan_results)
        QMessageBox.information(
            self, "Scan Summary",
            summary
        )
    
    def consolidate_files(self):
        """Consolidate selected files to output directory."""
        output_dir = QFileDialog.getExistingDirectory(
            self, "Select Output Directory", str(Path.home())
        )
        
        if not output_dir:
            return
        
        try:
            # Get selected images from database
            selected_images = self.db_manager.get_selected_images()
            
            if not selected_images:
                QMessageBox.warning(
                    self, "No Selected Images",
                    "No images have been selected for consolidation. "
                    "Please review duplicates first."
                )
                return
            
            # Show consolidation dialog
            dialog = ConsolidationDialog(self)
            if dialog.exec() == QDialog.DialogCode.Accepted:
                # TODO: Implement file consolidation
                QMessageBox.information(
                    self, "Consolidation",
                    f"Consolidated {len(selected_images)} files to {output_dir}"
                )
        
        except Exception as e:
            logger.error(f"Failed to consolidate files: {e}")
            QMessageBox.critical(
                self, "Error",
                f"Failed to consolidate files: {e}"
            )
    
    def test_similarity_detection(self):
        """Test similarity detection with sample images."""
        # TODO: Implement similarity detection test
        QMessageBox.information(
            self, "Test Similarity Detection",
            "This feature will be implemented in a future version."
        )
    
    def show_processing_stats(self):
        """Show processing statistics."""
        try:
            stats = self.db_manager.get_session_statistics()
            
            stats_text = f"""
Processing Statistics:

Total Images: {stats['total_images']}
Processed Images: {stats['processed_images']}
Duplicate Groups: {stats['duplicate_groups']}
Images in Duplicate Groups: {stats['duplicate_images']}
Selected Images: {stats['selected_images']}

Latest Scan:
"""
            
            if stats['latest_scan']:
                scan = stats['latest_scan']
                stats_text += f"""
Directories Scanned: {scan['directories_scanned']}
Total Images Found: {scan['total_images_found']}
Total Images Processed: {scan['total_images_processed']}
Corrupted Files Skipped: {scan['corrupted_files_skipped']}
Skipped Files: {scan['skipped_files_count']}
Scan Duration: {scan['scan_duration']:.2f} seconds
"""
            else:
                stats_text += "No scan data available"
            
            QMessageBox.information(
                self, "Processing Statistics",
                stats_text
            )
            
        except Exception as e:
            logger.error(f"Failed to show statistics: {e}")
            QMessageBox.critical(
                self, "Error",
                f"Failed to show statistics: {e}"
            )
    
    def clear_database(self):
        """Clear the database."""
        reply = QMessageBox.question(
            self, "Clear Database",
            "This will permanently delete all scanned images and duplicate groups.\n\n"
            "Are you sure you want to continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            try:
                self.db_manager.clear_database()
                self.consolidate_button.setEnabled(False)
                self.log_message("Database cleared")
                QMessageBox.information(
                    self, "Database Cleared",
                    "All data has been cleared from the database."
                )
            except Exception as e:
                logger.error(f"Failed to clear database: {e}")
                QMessageBox.critical(
                    self, "Error",
                    f"Failed to clear database: {e}"
                )


class ConsolidationDialog(QDialog):
    """Dialog for file consolidation options."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Consolidate Files")
        self.setModal(True)
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the dialog UI."""
        layout = QVBoxLayout(self)
        
        # Options group
        options_group = QGroupBox("Consolidation Options")
        options_layout = QVBoxLayout(options_group)
        
        # File organization
        org_layout = QHBoxLayout()
        org_layout.addWidget(QLabel("File Organization:"))
        self.org_combo = QComboBox()
        self.org_combo.addItems([
            "Flat structure (all files in one directory)",
            "Preserve directory structure",
            "Group by duplicate sets",
            "Group by quality score"
        ])
        org_layout.addWidget(self.org_combo)
        options_layout.addLayout(org_layout)
        
        # Naming strategy
        naming_layout = QHBoxLayout()
        naming_layout.addWidget(QLabel("File Naming:"))
        self.naming_combo = QComboBox()
        self.naming_combo.addItems([
            "Keep original names",
            "Add quality score prefix",
            "Add duplicate group prefix",
            "Use hash-based names"
        ])
        naming_layout.addWidget(self.naming_combo)
        options_layout.addLayout(naming_layout)
        
        # Conflict resolution
        conflict_layout = QHBoxLayout()
        conflict_layout.addWidget(QLabel("Name Conflicts:"))
        self.conflict_combo = QComboBox()
        self.conflict_combo.addItems([
            "Skip existing files",
            "Overwrite existing files",
            "Add suffix to duplicates",
            "Ask for each conflict"
        ])
        conflict_layout.addWidget(self.conflict_combo)
        options_layout.addLayout(conflict_layout)
        
        # Additional options
        self.preserve_metadata_checkbox = QCheckBox("Preserve file metadata")
        self.preserve_metadata_checkbox.setChecked(True)
        options_layout.addWidget(self.preserve_metadata_checkbox)
        
        self.create_log_checkbox = QCheckBox("Create consolidation log")
        self.create_log_checkbox.setChecked(True)
        options_layout.addWidget(self.create_log_checkbox)
        
        layout.addWidget(options_group)
        
        # Buttons
        buttons_layout = QHBoxLayout()
        buttons_layout.addStretch()
        
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        buttons_layout.addWidget(self.cancel_button)
        
        self.ok_button = QPushButton("Start Consolidation")
        self.ok_button.clicked.connect(self.accept)
        self.ok_button.setDefault(True)
        buttons_layout.addWidget(self.ok_button)
        
        layout.addLayout(buttons_layout)
