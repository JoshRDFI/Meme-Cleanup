"""
Logs tab for Meme-Cleanup.

Displays application logs and provides filtering and export functionality.
"""

import logging
import os
from pathlib import Path
from typing import List, Optional
from datetime import datetime

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QTextEdit,
    QComboBox, QCheckBox, QFileDialog, QMessageBox, QGroupBox, QLineEdit
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QFont, QTextCursor

from utils.config_manager import ConfigManager


logger = logging.getLogger(__name__)


class LogHandler(logging.Handler):
    """Custom log handler for GUI display."""
    
    def __init__(self, logs_tab):
        super().__init__()
        self.logs_tab = logs_tab
        self.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    
    def emit(self, record):
        """Emit log record to GUI."""
        try:
            msg = self.format(record)
            self.logs_tab.add_log_message(msg, record.levelname)
        except Exception:
            self.handleError(record)


class LogsTab(QWidget):
    """Tab for displaying application logs."""
    
    def __init__(self, config_manager: Optional[ConfigManager] = None):
        """
        Initialize logs tab.
        
        Args:
            config_manager: Configuration manager instance (optional)
        """
        super().__init__()
        self.config_manager = config_manager
        self.log_messages = []
        self.filtered_messages = []
        self.current_filter = "All"
        self.search_text = ""
        self.auto_scroll = True
        
        self.setup_ui()
        self.setup_log_handler()
        self.setup_timer()
        
        logger.info("Logs tab initialized")
    
    def setup_ui(self):
        """Setup the user interface."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        
        # Title
        title_label = QLabel("Application Logs")
        title_font = QFont()
        title_font.setPointSize(16)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setStyleSheet("color: #4A90E2; margin-bottom: 10px;")
        layout.addWidget(title_label)
        
        # Control panel
        control_group = QGroupBox("Log Controls")
        control_layout = QVBoxLayout(control_group)
        
        # Top row - filters and search
        top_row = QHBoxLayout()
        
        # Log level filter
        top_row.addWidget(QLabel("Log Level:"))
        self.log_level_combo = QComboBox()
        self.log_level_combo.addItems(["All", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
        self.log_level_combo.currentTextChanged.connect(self.apply_filters)
        top_row.addWidget(self.log_level_combo)
        
        # Search box
        top_row.addWidget(QLabel("Search:"))
        self.search_box = QLineEdit()
        self.search_box.setPlaceholderText("Search log messages...")
        self.search_box.textChanged.connect(self.apply_filters)
        top_row.addWidget(self.search_box)
        
        # Auto-scroll checkbox
        self.auto_scroll_checkbox = QCheckBox("Auto-scroll")
        self.auto_scroll_checkbox.setChecked(True)
        self.auto_scroll_checkbox.toggled.connect(self.toggle_auto_scroll)
        top_row.addWidget(self.auto_scroll_checkbox)
        
        top_row.addStretch()
        control_layout.addLayout(top_row)
        
        # Bottom row - action buttons
        bottom_row = QHBoxLayout()
        
        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.clicked.connect(self.refresh_logs)
        bottom_row.addWidget(self.refresh_button)
        
        self.clear_button = QPushButton("Clear Logs")
        self.clear_button.clicked.connect(self.clear_logs)
        bottom_row.addWidget(self.clear_button)
        
        self.save_button = QPushButton("Save Logs")
        self.save_button.clicked.connect(self.save_logs)
        bottom_row.addWidget(self.save_button)
        
        self.copy_button = QPushButton("Copy Selected")
        self.copy_button.clicked.connect(self.copy_selected)
        bottom_row.addWidget(self.copy_button)
        
        bottom_row.addStretch()
        
        # Log statistics
        self.stats_label = QLabel("Messages: 0 | Filtered: 0")
        self.stats_label.setStyleSheet("color: #666; font-style: italic;")
        bottom_row.addWidget(self.stats_label)
        
        control_layout.addLayout(bottom_row)
        layout.addWidget(control_group)
        
        # Log display
        self.log_display = QTextEdit()
        self.log_display.setReadOnly(True)
        self.log_display.setFont(QFont("Consolas", 9))
        self.log_display.setStyleSheet("""
            QTextEdit {
                background-color: #1E1E1E;
                color: #F5F5F5;
                border: 1px solid #2D2D2D;
                border-radius: 4px;
            }
        """)
        layout.addWidget(self.log_display)
        
        # Load existing logs
        self.load_existing_logs()
    
    def setup_log_handler(self):
        """Setup custom log handler for GUI display."""
        self.log_handler = LogHandler(self)
        logging.getLogger().addHandler(self.log_handler)
    
    def setup_timer(self):
        """Setup timer for periodic log updates."""
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_display)
        self.update_timer.start(1000)  # Update every second
    
    def add_log_message(self, message: str, level: str = "INFO"):
        """Add a log message to the display."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {level}: {message}"
        
        self.log_messages.append({
            'message': formatted_message,
            'level': level,
            'timestamp': timestamp,
            'raw_message': message
        })
        
        # Apply current filters
        self.apply_filters()
    
    def apply_filters(self):
        """Apply current filters to log messages."""
        level_filter = self.log_level_combo.currentText()
        search_text = self.search_box.text().lower()
        
        self.filtered_messages = []
        
        for msg in self.log_messages:
            # Apply level filter
            if level_filter != "All" and msg['level'] != level_filter:
                continue
            
            # Apply search filter
            if search_text and search_text not in msg['raw_message'].lower():
                continue
            
            self.filtered_messages.append(msg)
        
        self.update_display()
        self.update_stats()
    
    def update_display(self):
        """Update the log display with filtered messages."""
        if not self.filtered_messages:
            self.log_display.clear()
            return
        
        # Get current scroll position
        scrollbar = self.log_display.verticalScrollBar()
        was_at_bottom = scrollbar.value() == scrollbar.maximum()
        
        # Update display
        self.log_display.clear()
        for msg in self.filtered_messages:
            color = self.get_level_color(msg['level'])
            self.log_display.append(f'<span style="color: {color};">{msg["message"]}</span>')
        
        # Restore scroll position
        if self.auto_scroll and was_at_bottom:
            scrollbar.setValue(scrollbar.maximum())
    
    def get_level_color(self, level: str) -> str:
        """Get color for log level."""
        colors = {
            'DEBUG': '#888888',
            'INFO': '#4A90E2',
            'WARNING': '#F39C12',
            'ERROR': '#E74C3C',
            'CRITICAL': '#C0392B'
        }
        return colors.get(level, '#F5F5F5')
    
    def update_stats(self):
        """Update statistics display."""
        total = len(self.log_messages)
        filtered = len(self.filtered_messages)
        self.stats_label.setText(f"Messages: {total} | Filtered: {filtered}")
    
    def toggle_auto_scroll(self, enabled: bool):
        """Toggle auto-scroll functionality."""
        self.auto_scroll = enabled
    
    def refresh_logs(self):
        """Refresh log display."""
        self.apply_filters()
    
    def clear_logs(self):
        """Clear all log messages."""
        reply = QMessageBox.question(
            self, "Clear Logs",
            "Are you sure you want to clear all log messages?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            self.log_messages.clear()
            self.filtered_messages.clear()
            self.log_display.clear()
            self.update_stats()
    
    def save_logs(self):
        """Save logs to file."""
        if not self.filtered_messages:
            QMessageBox.warning(
                self, "No Logs",
                "No log messages to save."
            )
            return
        
        # Get save location
        default_path = ""
        if self.config_manager:
            default_path = str(self.config_manager.get_log_directory() / "exported_logs.txt")
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Logs",
            default_path,
            "Text Files (*.txt);;Log Files (*.log);;All Files (*)"
        )
        
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    for msg in self.filtered_messages:
                        f.write(f"{msg['message']}\n")
                
                QMessageBox.information(
                    self, "Logs Saved",
                    f"Logs have been saved to:\n{file_path}"
                )
                
            except Exception as e:
                logger.error(f"Failed to save logs: {e}")
                QMessageBox.critical(
                    self, "Error",
                    f"Failed to save logs: {e}"
                )
    
    def copy_selected(self):
        """Copy selected text to clipboard."""
        cursor = self.log_display.textCursor()
        if cursor.hasSelection():
            text = cursor.selectedText()
            self.log_display.copy()
            QMessageBox.information(
                self, "Copied",
                "Selected text has been copied to clipboard."
            )
        else:
            QMessageBox.warning(
                self, "No Selection",
                "Please select text to copy."
            )
    
    def load_existing_logs(self):
        """Load existing logs from file if available."""
        if not self.config_manager:
            return
        
        log_file = self.config_manager.get_log_file_path()
        if log_file and log_file.exists():
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    for line in f.readlines()[-100:]:  # Load last 100 lines
                        line = line.strip()
                        if line:
                            # Parse log line
                            try:
                                # Simple parsing - extract level and message
                                if ' - ' in line:
                                    parts = line.split(' - ', 3)
                                    if len(parts) >= 4:
                                        timestamp, name, level, message = parts
                                        self.add_log_message(message, level)
                                    else:
                                        self.add_log_message(line)
                                else:
                                    self.add_log_message(line)
                            except Exception:
                                self.add_log_message(line)
                
                logger.info(f"Loaded {len(self.log_messages)} existing log messages")
                
            except Exception as e:
                logger.error(f"Failed to load existing logs: {e}")
    
    def closeEvent(self, event):
        """Handle close event."""
        try:
            # Remove log handler
            if hasattr(self, 'log_handler'):
                logging.getLogger().removeHandler(self.log_handler)
            
            # Stop timer
            if hasattr(self, 'update_timer'):
                self.update_timer.stop()
            
        except Exception as e:
            logger.error(f"Error during logs tab cleanup: {e}")
        
        event.accept()
