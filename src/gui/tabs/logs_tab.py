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