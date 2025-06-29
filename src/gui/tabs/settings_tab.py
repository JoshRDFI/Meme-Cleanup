<<<<<<< HEAD
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
    QMessageBox, QFormLayout, QTabWidget, QTextEdit, QFrame
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
        self.quality_metric.addItems(["combined", "brisque", "niqe"])
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
        self.db_path_edit = QLineEdit()
        self.db_path_edit.setPlaceholderText("Leave empty for default location")
        self.db_path_edit.setToolTip("Custom database file location")
        db_layout.addWidget(self.db_path_edit)
        
        self.db_browse_button = QPushButton("Browse")
        self.db_browse_button.clicked.connect(self.browse_database_path)
        db_layout.addWidget(self.db_browse_button)
        
        paths_layout.addRow("Database Path:", db_layout)
        
        # Log file path
        log_layout = QHBoxLayout()
        self.log_path_edit = QLineEdit()
        self.log_path_edit.setPlaceholderText("Leave empty for default location")
        self.log_path_edit.setToolTip("Custom log file location")
        log_layout.addWidget(self.log_path_edit)
        
        self.log_browse_button = QPushButton("Browse")
        self.log_browse_button.clicked.connect(self.browse_log_path)
        log_layout.addWidget(self.log_browse_button)
        
        paths_layout.addRow("Log File Path:", log_layout)
        
        # Output directory
        output_layout = QHBoxLayout()
        self.output_dir_edit = QLineEdit()
        self.output_dir_edit.setPlaceholderText("Default output directory for consolidated images")
        self.output_dir_edit.setToolTip("Default directory for saving consolidated images")
        output_layout.addWidget(self.output_dir_edit)
        
        self.output_browse_button = QPushButton("Browse")
        self.output_browse_button.clicked.connect(self.browse_output_directory)
        output_layout.addWidget(self.output_browse_button)
        
        paths_layout.addRow("Output Directory:", output_layout)
        
        layout.addWidget(paths_group)
        layout.addStretch()
        
        return tab
    
    def create_sessions_tab(self) -> QWidget:
        """Create the sessions management tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        
        # Sessions group
        sessions_group = QGroupBox("Session Management")
        sessions_layout = QVBoxLayout(sessions_group)
        
        # Sessions list
        self.sessions_text = QTextEdit()
        self.sessions_text.setReadOnly(True)
        self.sessions_text.setMaximumHeight(200)
        self.sessions_text.setToolTip("List of available sessions")
        sessions_layout.addWidget(QLabel("Available Sessions:"))
        sessions_layout.addWidget(self.sessions_text)
        
        # Session actions
        session_buttons_layout = QHBoxLayout()
        
        self.refresh_sessions_button = QPushButton("Refresh Sessions")
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
            
            # UI settings
            self.dark_theme.setChecked(self.config_manager.ui.dark_theme)
            self.auto_save_interval.setValue(self.config_manager.ui.auto_save_interval)
            self.window_width.setValue(self.config_manager.ui.window_width)
            self.window_height.setValue(self.config_manager.ui.window_height)
            
            # Paths settings
            self.db_path_edit.setText(self.config_manager.paths.database_path)
            self.log_path_edit.setText(self.config_manager.paths.log_file_path)
            self.output_dir_edit.setText(self.config_manager.paths.output_directory)
            
            # Load sessions
            self.refresh_sessions()
            
            logger.info("Settings loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load settings: {e}")
    
    def save_settings(self):
        """Save current settings to configuration manager."""
        try:
            # Update processing config
            self.config_manager.update_processing_config(
                similarity_threshold=self.similarity_threshold.value(),
                batch_size=self.batch_size.value(),
                quality_metric=self.quality_metric.currentText(),
                use_gpu=self.use_gpu.isChecked(),
                parallel_processing=self.parallel_processing.isChecked(),
                n_jobs=self.n_jobs.value()
            )
            
            # Update UI config
            self.config_manager.update_ui_config(
                dark_theme=self.dark_theme.isChecked(),
                auto_save_interval=self.auto_save_interval.value(),
                window_width=self.window_width.value(),
                window_height=self.window_height.value()
            )
            
            # Update paths config
            self.config_manager.update_paths_config(
                database_path=self.db_path_edit.text(),
                log_file_path=self.log_path_edit.text(),
                output_directory=self.output_dir_edit.text()
            )
            
            QMessageBox.information(self, "Settings Saved", "Settings have been saved successfully.")
            logger.info("Settings saved successfully")
            
        except Exception as e:
            logger.error(f"Failed to save settings: {e}")
            QMessageBox.critical(self, "Error", f"Failed to save settings: {e}")
    
    def reset_settings(self):
        """Reset settings to default values."""
        reply = QMessageBox.question(
            self, "Reset Settings",
            "Are you sure you want to reset all settings to default values?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            try:
                self.config_manager.reset_config()
                self.load_settings()
                QMessageBox.information(self, "Settings Reset", "Settings have been reset to default values.")
                logger.info("Settings reset to defaults")
                
            except Exception as e:
                logger.error(f"Failed to reset settings: {e}")
                QMessageBox.critical(self, "Error", f"Failed to reset settings: {e}")
    
    def browse_database_path(self):
        """Browse for database file path."""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Select Database File", "",
            "SQLite Database (*.db);;All Files (*)"
        )
        if file_path:
            self.db_path_edit.setText(file_path)
    
    def browse_log_path(self):
        """Browse for log file path."""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Select Log File", "",
            "Log Files (*.log);;Text Files (*.txt);;All Files (*)"
        )
        if file_path:
            self.log_path_edit.setText(file_path)
    
    def browse_output_directory(self):
        """Browse for output directory."""
        dir_path = QFileDialog.getExistingDirectory(
            self, "Select Output Directory", ""
        )
        if dir_path:
            self.output_dir_edit.setText(dir_path)
    
    def refresh_sessions(self):
        """Refresh the sessions list."""
        try:
            sessions = self.config_manager.list_sessions()
            
            if not sessions:
                self.sessions_text.setText("No sessions found.")
                return
            
            # Format sessions for display
            session_text = ""
            for session in sessions:
                session_text += f"ID: {session['id']}\n"
                session_text += f"Name: {session['name']}\n"
                session_text += f"Status: {session['status']}\n"
                session_text += f"Images: {session['processed_images']}/{session['total_images']}\n"
                session_text += f"Last Modified: {session['last_modified'][:19]}\n"
                session_text += "-" * 40 + "\n"
            
            self.sessions_text.setText(session_text)
            
        except Exception as e:
            logger.error(f"Failed to refresh sessions: {e}")
            self.sessions_text.setText(f"Error loading sessions: {e}")
    
    def delete_selected_session(self):
        """Delete the selected session."""
        # For now, this is a placeholder. In a full implementation,
        # you would get the selected session from the text widget
        # and delete it. This would require a more sophisticated UI.
        QMessageBox.information(
            self, "Delete Session",
            "Session deletion functionality will be implemented in a future update."
        ) 
=======
"""
Settings tab for Meme-Cleanup.

Provides configuration options for the application.
"""

import logging
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QGroupBox,
    QCheckBox, QSpinBox, QDoubleSpinBox, QComboBox, QLineEdit, QFileDialog
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont


logger = logging.getLogger(__name__)


class SettingsTab(QWidget):
    """Tab for application settings."""
    
    def __init__(self):
        """Initialize settings tab."""
        super().__init__()
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
        
        # Processing settings
        processing_group = QGroupBox("Processing Settings")
        processing_layout = QVBoxLayout(processing_group)
        
        # Default similarity threshold
        threshold_layout = QHBoxLayout()
        threshold_layout.addWidget(QLabel("Default Similarity Threshold:"))
        self.default_similarity = QDoubleSpinBox()
        self.default_similarity.setRange(0.1, 1.0)
        self.default_similarity.setValue(0.8)
        self.default_similarity.setSingleStep(0.05)
        self.default_similarity.setDecimals(2)
        threshold_layout.addWidget(self.default_similarity)
        threshold_layout.addStretch()
        processing_layout.addLayout(threshold_layout)
        
        # Default batch size
        batch_layout = QHBoxLayout()
        batch_layout.addWidget(QLabel("Default Batch Size:"))
        self.default_batch_size = QSpinBox()
        self.default_batch_size.setRange(1, 32)
        self.default_batch_size.setValue(8)
        batch_layout.addWidget(self.default_batch_size)
        batch_layout.addStretch()
        processing_layout.addLayout(batch_layout)
        
        # Default quality metric
        quality_layout = QHBoxLayout()
        quality_layout.addWidget(QLabel("Default Quality Metric:"))
        self.default_quality_metric = QComboBox()
        self.default_quality_metric.addItems(["combined", "brisque", "niqe"])
        quality_layout.addWidget(self.default_quality_metric)
        quality_layout.addStretch()
        processing_layout.addLayout(quality_layout)
        
        # Use GPU by default
        self.use_gpu_default = QCheckBox("Use GPU by default")
        self.use_gpu_default.setChecked(True)
        processing_layout.addWidget(self.use_gpu_default)
        
        layout.addWidget(processing_group)
        
        # UI settings
        ui_group = QGroupBox("Interface Settings")
        ui_layout = QVBoxLayout(ui_group)
        
        # Auto-refresh interval
        refresh_layout = QHBoxLayout()
        refresh_layout.addWidget(QLabel("Auto-refresh Interval (seconds):"))
        self.auto_refresh_interval = QSpinBox()
        self.auto_refresh_interval.setRange(1, 60)
        self.auto_refresh_interval.setValue(5)
        refresh_layout.addWidget(self.auto_refresh_interval)
        refresh_layout.addStretch()
        ui_layout.addLayout(refresh_layout)
        
        # Show file paths
        self.show_file_paths = QCheckBox("Show full file paths in tables")
        self.show_file_paths.setChecked(False)
        ui_layout.addWidget(self.show_file_paths)
        
        # Confirm deletions
        self.confirm_deletions = QCheckBox("Confirm before deleting files")
        self.confirm_deletions.setChecked(True)
        ui_layout.addWidget(self.confirm_deletions)
        
        layout.addWidget(ui_group)
        
        # Storage settings
        storage_group = QGroupBox("Storage Settings")
        storage_layout = QVBoxLayout(storage_group)
        
        # Database location
        db_layout = QHBoxLayout()
        db_layout.addWidget(QLabel("Database Location:"))
        self.db_location = QLineEdit()
        self.db_location.setReadOnly(True)
        db_layout.addWidget(self.db_location)
        
        self.browse_db_button = QPushButton("Browse")
        self.browse_db_button.clicked.connect(self.browse_database_location)
        db_layout.addWidget(self.browse_db_button)
        
        storage_layout.addLayout(db_layout)
        
        # Log file location
        log_layout = QHBoxLayout()
        log_layout.addWidget(QLabel("Log File Location:"))
        self.log_location = QLineEdit()
        self.log_location.setReadOnly(True)
        log_layout.addWidget(self.log_location)
        
        self.browse_log_button = QPushButton("Browse")
        self.browse_log_button.clicked.connect(self.browse_log_location)
        log_layout.addWidget(self.browse_log_button)
        
        storage_layout.addLayout(log_layout)
        
        layout.addWidget(storage_group)
        
        # Action buttons
        action_layout = QHBoxLayout()
        
        self.save_button = QPushButton("Save Settings")
        self.save_button.clicked.connect(self.save_settings)
        action_layout.addWidget(self.save_button)
        
        self.reset_button = QPushButton("Reset to Defaults")
        self.reset_button.clicked.connect(self.reset_settings)
        action_layout.addWidget(self.reset_button)
        
        action_layout.addStretch()
        
        layout.addLayout(action_layout)
        layout.addStretch()
    
    def load_settings(self):
        """Load current settings."""
        # TODO: Load settings from configuration file
        logger.info("Loading settings")
    
    def save_settings(self):
        """Save current settings."""
        # TODO: Save settings to configuration file
        logger.info("Saving settings")
    
    def reset_settings(self):
        """Reset settings to defaults."""
        # TODO: Reset all settings to default values
        logger.info("Resetting settings to defaults")
    
    def browse_database_location(self):
        """Browse for database location."""
        # TODO: Implement database location browsing
        logger.info("Browsing for database location")
    
    def browse_log_location(self):
        """Browse for log file location."""
        # TODO: Implement log file location browsing
        logger.info("Browsing for log file location") 
>>>>>>> 88a757b8ecc4ae4d819ae02a34c56b2a1ebc3714
