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