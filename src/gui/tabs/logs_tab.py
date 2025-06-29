<<<<<<< HEAD
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
            # Color code by level
            color = self.get_level_color(msg['level'])
            self.log_display.append(f'<span style="color: {color};">{msg["message"]}</span>')
        
        # Restore scroll position
        if self.auto_scroll and was_at_bottom:
            cursor = self.log_display.textCursor()
            cursor.movePosition(QTextCursor.MoveOperation.End)
            self.log_display.setTextCursor(cursor)
    
    def get_level_color(self, level: str) -> str:
        """Get color for log level."""
        colors = {
            'DEBUG': '#888888',
            'INFO': '#4A90E2',
            'WARNING': '#FFA500',
            'ERROR': '#E94E77',
            'CRITICAL': '#FF0000'
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
        """Refresh the log display."""
        self.apply_filters()
        logger.info("Logs refreshed")
    
    def clear_logs(self):
        """Clear all log messages."""
        reply = QMessageBox.question(
            self, "Clear Logs",
            "Are you sure you want to clear all log messages?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            self.log_messages.clear()
            self.filtered_messages.clear()
            self.log_display.clear()
            self.update_stats()
            logger.info("Logs cleared")
    
    def save_logs(self):
        """Save logs to file."""
        if not self.filtered_messages:
            QMessageBox.information(
                self, "No Logs",
                "No log messages to save."
            )
            return
        
        # Get save location
        if self.config_manager and self.config_manager.paths.log_file_path:
            default_path = self.config_manager.paths.log_file_path
        else:
            default_path = str(Path.home() / "meme_cleanup_logs.txt")
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Logs",
            default_path,
            "Text Files (*.txt);;Log Files (*.log);;All Files (*)"
        )
        
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(f"Meme-Cleanup Logs - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write("=" * 60 + "\n\n")
                    
                    for msg in self.filtered_messages:
                        f.write(f"{msg['message']}\n")
                
                QMessageBox.information(
                    self, "Logs Saved",
                    f"Logs saved successfully to:\n{file_path}"
                )
                
                logger.info(f"Logs saved to: {file_path}")
                
            except Exception as e:
                logger.error(f"Failed to save logs: {e}")
                QMessageBox.critical(
                    self, "Save Error",
                    f"Failed to save logs: {e}"
                )
    
    def copy_selected(self):
        """Copy selected text to clipboard."""
        cursor = self.log_display.textCursor()
        selected_text = cursor.selectedText()
        
        if selected_text:
            clipboard = self.log_display.clipboard()
            clipboard.setText(selected_text)
            QMessageBox.information(
                self, "Copied",
                "Selected text copied to clipboard."
            )
        else:
            QMessageBox.information(
                self, "No Selection",
                "Please select text to copy."
            )
    
    def load_existing_logs(self):
        """Load existing logs from log file."""
        if not self.config_manager:
            return
        
        log_file_path = self.config_manager.get_log_file_path()
        if not log_file_path.exists():
            return
        
        try:
            with open(log_file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Parse existing log lines (simple parsing)
            for line in lines[-100:]:  # Load last 100 lines
                line = line.strip()
                if line:
                    # Try to parse log level from line
                    level = "INFO"
                    if " - DEBUG - " in line:
                        level = "DEBUG"
                    elif " - INFO - " in line:
                        level = "INFO"
                    elif " - WARNING - " in line:
                        level = "WARNING"
                    elif " - ERROR - " in line:
                        level = "ERROR"
                    elif " - CRITICAL - " in line:
                        level = "CRITICAL"
                    
                    # Extract message
                    parts = line.split(" - ", 3)
                    if len(parts) >= 4:
                        message = parts[3]
                    else:
                        message = line
                    
                    self.add_log_message(message, level)
            
            logger.info(f"Loaded {len(lines)} existing log lines")
            
        except Exception as e:
            logger.error(f"Failed to load existing logs: {e}")
    
    def closeEvent(self, event):
        """Handle tab close event."""
        # Remove custom log handler
        if hasattr(self, 'log_handler'):
            logging.getLogger().removeHandler(self.log_handler)
        
        event.accept() 
=======
"""
Logs tab for Meme-Cleanup.

Displays application logs and processing information.
"""

import logging
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QTextEdit,
    QComboBox, QCheckBox, QGroupBox
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont, QTextCursor


logger = logging.getLogger(__name__)


class LogsTab(QWidget):
    """Tab for displaying application logs."""
    
    def __init__(self):
        """Initialize logs tab."""
        super().__init__()
        self.setup_ui()
        self.setup_log_capture()
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
        control_layout = QHBoxLayout()
        
        # Log level filter
        control_layout.addWidget(QLabel("Log Level:"))
        self.log_level_combo = QComboBox()
        self.log_level_combo.addItems(["All", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
        self.log_level_combo.setCurrentText("INFO")
        self.log_level_combo.currentTextChanged.connect(self.filter_logs)
        control_layout.addWidget(self.log_level_combo)
        
        # Auto-scroll
        self.auto_scroll_checkbox = QCheckBox("Auto-scroll")
        self.auto_scroll_checkbox.setChecked(True)
        control_layout.addWidget(self.auto_scroll_checkbox)
        
        # Clear logs
        self.clear_button = QPushButton("Clear Logs")
        self.clear_button.clicked.connect(self.clear_logs)
        control_layout.addWidget(self.clear_button)
        
        # Save logs
        self.save_button = QPushButton("Save Logs")
        self.save_button.clicked.connect(self.save_logs)
        control_layout.addWidget(self.save_button)
        
        control_layout.addStretch()
        
        layout.addLayout(control_layout)
        
        # Log display
        log_group = QGroupBox("Log Messages")
        log_layout = QVBoxLayout(log_group)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Consolas", 10))
        self.log_text.setStyleSheet("""
            QTextEdit {
                background-color: #1E1E1E;
                color: #F5F5F5;
                border: 1px solid #3D3D3D;
                border-radius: 4px;
            }
        """)
        log_layout.addWidget(self.log_text)
        
        layout.addWidget(log_group)
        
        # Status bar
        status_layout = QHBoxLayout()
        
        self.log_count_label = QLabel("Log entries: 0")
        status_layout.addWidget(self.log_count_label)
        
        self.last_update_label = QLabel("Last update: Never")
        status_layout.addWidget(self.last_update_label)
        
        status_layout.addStretch()
        
        layout.addLayout(status_layout)
    
    def setup_log_capture(self):
        """Setup log capture to display logs in the UI."""
        # Create a custom log handler
        self.log_handler = LogDisplayHandler(self)
        
        # Add handler to root logger
        root_logger = logging.getLogger()
        root_logger.addHandler(self.log_handler)
        
        # Add initial log message
        self.add_log_message("INFO", "Logs tab initialized and ready to capture messages.")
    
    def add_log_message(self, level: str, message: str):
        """Add a log message to the display."""
        # Check if message should be filtered
        if not self.should_display_message(level):
            return
        
        # Format the message
        formatted_message = f"[{level}] {message}"
        
        # Add to text widget
        self.log_text.append(formatted_message)
        
        # Auto-scroll if enabled
        if self.auto_scroll_checkbox.isChecked():
            cursor = self.log_text.textCursor()
            cursor.movePosition(QTextCursor.MoveOperation.End)
            self.log_text.setTextCursor(cursor)
        
        # Update status
        self.update_status()
    
    def should_display_message(self, level: str) -> bool:
        """Check if a message should be displayed based on current filter."""
        current_filter = self.log_level_combo.currentText()
        
        if current_filter == "All":
            return True
        
        # Define log level hierarchy
        levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        
        try:
            message_level_index = levels.index(level)
            filter_level_index = levels.index(current_filter)
            return message_level_index >= filter_level_index
        except ValueError:
            return True
    
    def filter_logs(self):
        """Filter logs based on current level selection."""
        # TODO: Implement log filtering
        logger.info("Log filtering not yet implemented")
    
    def clear_logs(self):
        """Clear all log messages."""
        self.log_text.clear()
        self.update_status()
        logger.info("Logs cleared by user")
    
    def save_logs(self):
        """Save logs to a file."""
        # TODO: Implement log saving
        logger.info("Log saving not yet implemented")
    
    def update_status(self):
        """Update status information."""
        # Count log entries
        text = self.log_text.toPlainText()
        line_count = len(text.split('\n')) if text else 0
        self.log_count_label.setText(f"Log entries: {line_count}")
        
        # Update last update time
        from datetime import datetime
        self.last_update_label.setText(f"Last update: {datetime.now().strftime('%H:%M:%S')}")


class LogDisplayHandler(logging.Handler):
    """Custom log handler to display logs in the UI."""
    
    def __init__(self, logs_tab: LogsTab):
        super().__init__()
        self.logs_tab = logs_tab
    
    def emit(self, record):
        """Emit a log record to the UI."""
        try:
            # Format the message
            message = self.format(record)
            
            # Add to UI (use QTimer to ensure thread safety)
            QTimer.singleShot(0, lambda: self.logs_tab.add_log_message(record.levelname, message))
            
        except Exception as e:
            # Fallback to stderr if UI logging fails
            import sys
            print(f"Log display error: {e}", file=sys.stderr) 
>>>>>>> 88a757b8ecc4ae4d819ae02a34c56b2a1ebc3714
