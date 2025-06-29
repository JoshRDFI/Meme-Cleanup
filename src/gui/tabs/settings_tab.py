"""
Settings tab for Meme-Cleanup.

Handles application configuration and user preferences.
"""

import logging
from pathlib import Path
from typing import Optional

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QGroupBox,
    QCheckBox, QSpinBox, QDoubleSpinBox, QComboBox, QLineEdit, QFileDialog,
    QMessageBox, QFormLayout, QTabWidget, QTextEdit, QFrame, QListWidget
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont

from utils.config_manager import ConfigManager


logger = logging.getLogger(__name__)


class SettingsTab(QWidget):
    """Tab for application settings and configuration."""
    
    def __init__(self, config_manager: ConfigManager):
        """
        Initialize settings tab.
        
        Args:
            config_manager: Configuration manager instance
        """
        super().__init__()
        self.config_manager = config_manager
        
        self.setup_ui()
        self.load_settings()
        logger.info("Settings tab initialized")
    
    def setup_ui(self):
        """Setup the user interface."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        
        # Title
        title_label = QLabel("Settings")
        title_font = QFont()
        title_font.setPointSize(16)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setStyleSheet("color: #4A90E2; margin-bottom: 10px;")
        layout.addWidget(title_label)
        
        # Tab widget for different setting categories
        self.tab_widget = QTabWidget()
        
        # Processing settings tab
        self.processing_tab = self.create_processing_tab()
        self.tab_widget.addTab(self.processing_tab, "Processing")
        
        # UI settings tab
        self.ui_tab = self.create_ui_tab()
        self.tab_widget.addTab(self.ui_tab, "Interface")
        
        # Paths settings tab
        self.paths_tab = self.create_paths_tab()
        self.tab_widget.addTab(self.paths_tab, "Paths")
        
        # Session management tab
        self.sessions_tab = self.create_sessions_tab()
        self.tab_widget.addTab(self.sessions_tab, "Sessions")
        
        layout.addWidget(self.tab_widget)
        
        # Action buttons
        button_layout = QHBoxLayout()
        
        self.save_button = QPushButton("Save Settings")
        self.save_button.clicked.connect(self.save_settings)
        button_layout.addWidget(self.save_button)
        
        self.reset_button = QPushButton("Reset to Defaults")
        self.reset_button.clicked.connect(self.reset_settings)
        button_layout.addWidget(self.reset_button)
        
        button_layout.addStretch()
        
        layout.addLayout(button_layout)
    
    def create_processing_tab(self) -> QWidget:
        """Create the processing settings tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        
        # Processing configuration group
        processing_group = QGroupBox("Processing Configuration")
        processing_layout = QFormLayout(processing_group)
        
        # Similarity threshold
        self.similarity_threshold = QDoubleSpinBox()
        self.similarity_threshold.setRange(0.1, 1.0)
        self.similarity_threshold.setValue(0.8)
        self.similarity_threshold.setSingleStep(0.05)
        self.similarity_threshold.setDecimals(2)
        self.similarity_threshold.setToolTip("Threshold for considering images as duplicates (0.1-1.0)")
        processing_layout.addRow("Similarity Threshold:", self.similarity_threshold)
        
        # Batch size
        self.batch_size = QSpinBox()
        self.batch_size.setRange(1, 64)
        self.batch_size.setValue(16)
        self.batch_size.setToolTip("Number of images processed simultaneously")
        processing_layout.addRow("Batch Size:", self.batch_size)
        
        # Quality metric
        self.quality_metric = QComboBox()
        self.quality_metric.addItems(["BRISQUE", "NIQE", "Combined"])
        self.quality_metric.setToolTip("Quality metric for selecting best images")
        processing_layout.addRow("Quality Metric:", self.quality_metric)
        
        # Use GPU
        self.use_gpu = QCheckBox("Use GPU (CUDA)")
        self.use_gpu.setChecked(True)
        self.use_gpu.setToolTip("Enable GPU acceleration for processing")
        processing_layout.addRow("", self.use_gpu)
        
        # Parallel processing
        self.parallel_processing = QCheckBox("Enable Parallel Processing")
        self.parallel_processing.setChecked(True)
        self.parallel_processing.setToolTip("Use multiple CPU cores for processing")
        processing_layout.addRow("", self.parallel_processing)
        
        # Number of jobs
        self.n_jobs = QSpinBox()
        self.n_jobs.setRange(-1, 32)
        self.n_jobs.setValue(-1)
        self.n_jobs.setToolTip("-1 = Use all available CPU cores")
        processing_layout.addRow("CPU Jobs:", self.n_jobs)
        
        # Skip options
        self.skip_corrupted = QCheckBox("Skip Corrupted Files")
        self.skip_corrupted.setChecked(True)
        self.skip_corrupted.setToolTip("Skip files that cannot be loaded")
        processing_layout.addRow("", self.skip_corrupted)
        
        self.skip_animated = QCheckBox("Skip Animated Files")
        self.skip_animated.setChecked(False)
        self.skip_animated.setToolTip("Skip GIF and animated WebP files")
        processing_layout.addRow("", self.skip_animated)
        
        # Min file size
        self.min_file_size = QSpinBox()
        self.min_file_size.setRange(0, 1000000)
        self.min_file_size.setValue(1000)
        self.min_file_size.setSuffix(" bytes")
        self.min_file_size.setToolTip("Minimum file size to process")
        processing_layout.addRow("Minimum File Size:", self.min_file_size)
        
        # Save progress
        self.save_progress = QCheckBox("Save Progress Automatically")
        self.save_progress.setChecked(True)
        self.save_progress.setToolTip("Automatically save processing progress")
        processing_layout.addRow("", self.save_progress)
        
        layout.addWidget(processing_group)
        layout.addStretch()
        
        return tab
    
    def create_ui_tab(self) -> QWidget:
        """Create the UI settings tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        
        # Interface configuration group
        ui_group = QGroupBox("Interface Configuration")
        ui_layout = QFormLayout(ui_group)
        
        # Dark theme
        self.dark_theme = QCheckBox("Use Dark Theme")
        self.dark_theme.setChecked(True)
        self.dark_theme.setToolTip("Enable dark theme for the application")
        ui_layout.addRow("", self.dark_theme)
        
        # Auto-save enabled
        self.auto_save_enabled = QCheckBox("Enable Auto-save")
        self.auto_save_enabled.setChecked(True)
        self.auto_save_enabled.setToolTip("Automatically save sessions")
        ui_layout.addRow("", self.auto_save_enabled)
        
        # Auto-save interval
        self.auto_save_interval = QSpinBox()
        self.auto_save_interval.setRange(30, 3600)
        self.auto_save_interval.setValue(300)
        self.auto_save_interval.setSuffix(" seconds")
        self.auto_save_interval.setToolTip("Interval for automatic session saving")
        ui_layout.addRow("Auto-save Interval:", self.auto_save_interval)
        
        # Window size
        window_size_layout = QHBoxLayout()
        
        self.window_width = QSpinBox()
        self.window_width.setRange(800, 2560)
        self.window_width.setValue(1200)
        self.window_width.setSuffix(" px")
        window_size_layout.addWidget(QLabel("Width:"))
        window_size_layout.addWidget(self.window_width)
        
        self.window_height = QSpinBox()
        self.window_height.setRange(600, 1440)
        self.window_height.setValue(800)
        self.window_height.setSuffix(" px")
        window_size_layout.addWidget(QLabel("Height:"))
        window_size_layout.addWidget(self.window_height)
        
        ui_layout.addRow("Default Window Size:", window_size_layout)
        
        layout.addWidget(ui_group)
        layout.addStretch()
        
        return tab
    
    def create_paths_tab(self) -> QWidget:
        """Create the paths settings tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        
        # Paths configuration group
        paths_group = QGroupBox("Path Configuration")
        paths_layout = QFormLayout(paths_group)
        
        # Database path
        db_layout = QHBoxLayout()
        self.db_path = QLineEdit()
        self.db_path.setToolTip("Path to SQLite database file")
        db_layout.addWidget(self.db_path)
        
        self.browse_db_button = QPushButton("Browse")
        self.browse_db_button.clicked.connect(self.browse_database_path)
        db_layout.addWidget(self.browse_db_button)
        
        paths_layout.addRow("Database Path:", db_layout)
        
        # Log file path
        log_layout = QHBoxLayout()
        self.log_path = QLineEdit()
        self.log_path.setToolTip("Path to log file")
        log_layout.addWidget(self.log_path)
        
        self.browse_log_button = QPushButton("Browse")
        self.browse_log_button.clicked.connect(self.browse_log_path)
        log_layout.addWidget(self.browse_log_button)
        
        paths_layout.addRow("Log File Path:", log_layout)
        
        # Output directory
        output_layout = QHBoxLayout()
        self.output_directory = QLineEdit()
        self.output_directory.setToolTip("Default output directory for consolidated files")
        output_layout.addWidget(self.output_directory)
        
        self.browse_output_button = QPushButton("Browse")
        self.browse_output_button.clicked.connect(self.browse_output_directory)
        output_layout.addWidget(self.browse_output_button)
        
        paths_layout.addRow("Output Directory:", output_layout)
        
        layout.addWidget(paths_group)
        layout.addStretch()
        
        return tab
    
    def create_sessions_tab(self) -> QWidget:
        """Create the session management tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        
        # Session management group
        sessions_group = QGroupBox("Session Management")
        sessions_layout = QVBoxLayout(sessions_group)
        
        # Session list
        self.session_list = QListWidget()
        self.session_list.setMaximumHeight(200)
        sessions_layout.addWidget(self.session_list)
        
        # Session buttons
        session_buttons_layout = QHBoxLayout()
        
        self.refresh_sessions_button = QPushButton("Refresh")
        self.refresh_sessions_button.clicked.connect(self.refresh_sessions)
        session_buttons_layout.addWidget(self.refresh_sessions_button)
        
        self.delete_session_button = QPushButton("Delete Selected")
        self.delete_session_button.clicked.connect(self.delete_selected_session)
        session_buttons_layout.addWidget(self.delete_session_button)
        
        session_buttons_layout.addStretch()
        sessions_layout.addLayout(session_buttons_layout)
        
        layout.addWidget(sessions_group)
        layout.addStretch()
        
        return tab
    
    def load_settings(self):
        """Load current settings from configuration manager."""
        try:
            # Processing settings
            self.similarity_threshold.setValue(self.config_manager.processing.similarity_threshold)
            self.batch_size.setValue(self.config_manager.processing.batch_size)
            self.quality_metric.setCurrentText(self.config_manager.processing.quality_metric)
            self.use_gpu.setChecked(self.config_manager.processing.use_gpu)
            self.parallel_processing.setChecked(self.config_manager.processing.parallel_processing)
            self.n_jobs.setValue(self.config_manager.processing.n_jobs)
            self.skip_corrupted.setChecked(self.config_manager.processing.skip_corrupted)
            self.skip_animated.setChecked(self.config_manager.processing.skip_animated)
            self.min_file_size.setValue(self.config_manager.processing.min_file_size)
            self.save_progress.setChecked(self.config_manager.processing.save_progress)
            
            # UI settings
            self.dark_theme.setChecked(self.config_manager.ui.dark_theme)
            self.auto_save_enabled.setChecked(self.config_manager.ui.auto_save_enabled)
            self.auto_save_interval.setValue(self.config_manager.ui.auto_save_interval)
            self.window_width.setValue(self.config_manager.ui.window_width)
            self.window_height.setValue(self.config_manager.ui.window_height)
            
            # Paths
            self.db_path.setText(str(self.config_manager.paths.database_path))
            self.log_path.setText(str(self.config_manager.paths.log_file_path))
            self.output_directory.setText(str(self.config_manager.paths.output_directory))
            
            # Refresh sessions
            self.refresh_sessions()
            
        except Exception as e:
            logger.error(f"Failed to load settings: {e}")
    
    def save_settings(self):
        """Save current settings to configuration manager."""
        try:
            # Processing settings
            self.config_manager.processing.similarity_threshold = self.similarity_threshold.value()
            self.config_manager.processing.batch_size = self.batch_size.value()
            self.config_manager.processing.quality_metric = self.quality_metric.currentText()
            self.config_manager.processing.use_gpu = self.use_gpu.isChecked()
            self.config_manager.processing.parallel_processing = self.parallel_processing.isChecked()
            self.config_manager.processing.n_jobs = self.n_jobs.value()
            self.config_manager.processing.skip_corrupted = self.skip_corrupted.isChecked()
            self.config_manager.processing.skip_animated = self.skip_animated.isChecked()
            self.config_manager.processing.min_file_size = self.min_file_size.value()
            self.config_manager.processing.save_progress = self.save_progress.isChecked()
            
            # UI settings
            self.config_manager.ui.dark_theme = self.dark_theme.isChecked()
            self.config_manager.ui.auto_save_enabled = self.auto_save_enabled.isChecked()
            self.config_manager.ui.auto_save_interval = self.auto_save_interval.value()
            self.config_manager.ui.window_width = self.window_width.value()
            self.config_manager.ui.window_height = self.window_height.value()
            
            # Paths
            self.config_manager.paths.database_path = Path(self.db_path.text())
            self.config_manager.paths.log_file_path = Path(self.log_path.text()) if self.log_path.text() else None
            self.config_manager.paths.output_directory = Path(self.output_directory.text())
            
            # Save to file
            self.config_manager.save_config()
            
            QMessageBox.information(
                self, "Settings Saved",
                "Settings have been saved successfully."
            )
            
        except Exception as e:
            logger.error(f"Failed to save settings: {e}")
            QMessageBox.critical(
                self, "Error",
                f"Failed to save settings: {e}"
            )
    
    def reset_settings(self):
        """Reset settings to defaults."""
        reply = QMessageBox.question(
            self, "Reset Settings",
            "This will reset all settings to their default values.\n\n"
            "Are you sure you want to continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            try:
                self.config_manager.reset_to_defaults()
                self.load_settings()
                QMessageBox.information(
                    self, "Settings Reset",
                    "Settings have been reset to defaults."
                )
            except Exception as e:
                logger.error(f"Failed to reset settings: {e}")
                QMessageBox.critical(
                    self, "Error",
                    f"Failed to reset settings: {e}"
                )
    
    def browse_database_path(self):
        """Browse for database path."""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Select Database File",
            str(self.config_manager.paths.database_path),
            "SQLite Database (*.db);;All Files (*)"
        )
        if file_path:
            self.db_path.setText(file_path)
    
    def browse_log_path(self):
        """Browse for log file path."""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Select Log File",
            str(self.config_manager.paths.log_file_path or Path.home()),
            "Log Files (*.log);;Text Files (*.txt);;All Files (*)"
        )
        if file_path:
            self.log_path.setText(file_path)
    
    def browse_output_directory(self):
        """Browse for output directory."""
        directory = QFileDialog.getExistingDirectory(
            self, "Select Output Directory",
            str(self.config_manager.paths.output_directory)
        )
        if directory:
            self.output_directory.setText(directory)
    
    def refresh_sessions(self):
        """Refresh the session list."""
        try:
            self.session_list.clear()
            sessions = self.config_manager.list_sessions()
            
            for session in sessions:
                session_info = f"{session['name']} ({session['created_at']})"
                self.session_list.addItem(session_info)
                
        except Exception as e:
            logger.error(f"Failed to refresh sessions: {e}")
    
    def delete_selected_session(self):
        """Delete the selected session."""
        current_row = self.session_list.currentRow()
        if current_row < 0:
            QMessageBox.warning(
                self, "No Selection",
                "Please select a session to delete."
            )
            return
        
        reply = QMessageBox.question(
            self, "Delete Session",
            "Are you sure you want to delete the selected session?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            try:
                sessions = self.config_manager.list_sessions()
                if current_row < len(sessions):
                    session = sessions[current_row]
                    self.config_manager.delete_session(session['id'])
                    self.refresh_sessions()
                    QMessageBox.information(
                        self, "Session Deleted",
                        "Session has been deleted successfully."
                    )
            except Exception as e:
                logger.error(f"Failed to delete session: {e}")
                QMessageBox.critical(
                    self, "Error",
                    f"Failed to delete session: {e}"
                )
