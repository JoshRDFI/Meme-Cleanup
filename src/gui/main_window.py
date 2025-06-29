"""
Main window for Meme-Cleanup application.

Contains the primary PyQt6 window with tabs for different functionality areas.
"""

import logging
from pathlib import Path
from typing import List, Optional

from PyQt6.QtWidgets import (
    QMainWindow, QTabWidget, QVBoxLayout, QHBoxLayout, QWidget,
    QPushButton, QLabel, QProgressBar, QStatusBar, QMenuBar,
    QFileDialog, QMessageBox, QSplitter, QFrame, QInputDialog
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QAction, QFont, QPalette, QColor

from db.database import DatabaseManager
from utils.config_manager import ConfigManager
from gui.tabs.scan_tab import ScanTab
from gui.tabs.review_tab import ReviewTab
from gui.tabs.settings_tab import SettingsTab
from gui.tabs.logs_tab import LogsTab
from gui.styles import apply_dark_theme


logger = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    """Main application window."""
    
    def __init__(self, db_manager: DatabaseManager, config_manager: ConfigManager):
        """
        Initialize main window.
        
        Args:
            db_manager: Database manager instance
            config_manager: Configuration manager instance
        """
        super().__init__()
        self.db_manager = db_manager
        self.config_manager = config_manager
        self.current_session_id = None
        
        self.setWindowTitle("Meme-Cleanup - Image Deduplication Tool")
        self.setMinimumSize(1200, 800)
        
        # Apply window size from config
        self.resize(self.config_manager.ui.window_width, self.config_manager.ui.window_height)
        
        # Apply dark theme if enabled
        if self.config_manager.ui.dark_theme:
            apply_dark_theme(self)
        
        # Setup UI
        self.setup_ui()
        self.setup_menu()
        self.setup_status_bar()
        
        # Setup timer for status updates and auto-save
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self.update_status)
        self.status_timer.start(1000)  # Update every second
        
        # Auto-save timer
        self.auto_save_timer = QTimer()
        self.auto_save_timer.timeout.connect(self.auto_save_session)
        self.auto_save_timer.start(self.config_manager.ui.auto_save_interval * 1000)
        
        logger.info("Main window initialized")
    
    def setup_ui(self):
        """Setup the main user interface."""
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        
        # Header
        header_layout = QHBoxLayout()
        
        # Title
        title_label = QLabel("Meme-Cleanup")
        title_font = QFont()
        title_font.setPointSize(18)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setStyleSheet("color: #4A90E2; margin-bottom: 10px;")
        
        header_layout.addWidget(title_label)
        header_layout.addStretch()
        
        # Session info
        self.session_label = QLabel("No active session")
        self.session_label.setStyleSheet("color: #666; font-style: italic;")
        header_layout.addWidget(self.session_label)
        
        # Quick action buttons
        self.scan_button = QPushButton("Scan Directories")
        self.scan_button.setStyleSheet("""
            QPushButton {
                background-color: #4A90E2;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #357ABD;
            }
            QPushButton:pressed {
                background-color: #2D5A8E;
            }
        """)
        self.scan_button.clicked.connect(self.quick_scan)
        
        self.review_button = QPushButton("Review Duplicates")
        self.review_button.setStyleSheet("""
            QPushButton {
                background-color: #E94E77;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #D13A63;
            }
            QPushButton:pressed {
                background-color: #B82E4F;
            }
        """)
        self.review_button.clicked.connect(self.quick_review)
        
        header_layout.addWidget(self.scan_button)
        header_layout.addWidget(self.review_button)
        
        main_layout.addLayout(header_layout)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #2D2D2D;
                border-radius: 5px;
                text-align: center;
                background-color: #1E1E1E;
            }
            QProgressBar::chunk {
                background-color: #4A90E2;
                border-radius: 3px;
            }
        """)
        main_layout.addWidget(self.progress_bar)
        
        # Tab widget
        self.tab_widget = QTabWidget()
        self.tab_widget.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #2D2D2D;
                background-color: #1E1E1E;
            }
            QTabBar::tab {
                background-color: #2D2D2D;
                color: #F5F5F5;
                padding: 8px 16px;
                margin-right: 2px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            QTabBar::tab:selected {
                background-color: #4A90E2;
                color: white;
            }
            QTabBar::tab:hover {
                background-color: #3A7ABD;
            }
        """)
        
        # Create tabs
        self.scan_tab = ScanTab(self.db_manager, self.config_manager)
        self.review_tab = ReviewTab(self.db_manager)
        self.settings_tab = SettingsTab(self.config_manager)
        self.logs_tab = LogsTab(self.config_manager)
        
        # Add tabs
        self.tab_widget.addTab(self.scan_tab, "Scan & Process")
        self.tab_widget.addTab(self.review_tab, "Review Duplicates")
        self.tab_widget.addTab(self.settings_tab, "Settings")
        self.tab_widget.addTab(self.logs_tab, "Logs")
        
        main_layout.addWidget(self.tab_widget)
        
        # Connect tab signals
        self.scan_tab.progress_updated.connect(self.update_progress)
        self.scan_tab.processing_finished.connect(self.on_processing_finished)
        self.scan_tab.processing_started.connect(self.on_processing_started)
    
    def setup_menu(self):
        """Setup the application menu bar."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("&File")
        
        # Session management
        new_session_action = QAction("&New Session", self)
        new_session_action.setShortcut("Ctrl+N")
        new_session_action.triggered.connect(self.new_session)
        file_menu.addAction(new_session_action)
        
        open_session_action = QAction("&Open Session", self)
        open_session_action.setShortcut("Ctrl+O")
        open_session_action.triggered.connect(self.open_session)
        file_menu.addAction(open_session_action)
        
        save_session_action = QAction("&Save Session", self)
        save_session_action.setShortcut("Ctrl+S")
        save_session_action.triggered.connect(self.save_session)
        file_menu.addAction(save_session_action)
        
        file_menu.addSeparator()
        
        # Clear database
        clear_db_action = QAction("&Clear Database", self)
        clear_db_action.triggered.connect(self.clear_database)
        file_menu.addAction(clear_db_action)
        
        file_menu.addSeparator()
        
        # Exit
        exit_action = QAction("E&xit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Help menu
        help_menu = menubar.addMenu("&Help")
        
        about_action = QAction("&About", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
    
    def setup_status_bar(self):
        """Setup the status bar."""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        # Status labels
        self.status_label = QLabel("Ready")
        self.status_bar.addWidget(self.status_label)
        
        self.status_bar.addPermanentWidget(QLabel("|"))
        
        self.db_status_label = QLabel("Database: Connected")
        self.status_bar.addPermanentWidget(self.db_status_label)
        
        self.status_bar.addPermanentWidget(QLabel("|"))
        
        self.session_status_label = QLabel("Session: None")
        self.status_bar.addPermanentWidget(self.session_status_label)
    
    def quick_scan(self):
        """Quick scan action from header button."""
        self.tab_widget.setCurrentIndex(0)
        self.scan_tab.start_scan()
    
    def quick_review(self):
        """Quick review action from header button."""
        self.tab_widget.setCurrentIndex(1)
        self.review_tab.refresh_duplicates()
    
    def update_progress(self, value: int, maximum: int, message: str):
        """Update progress bar and status."""
        self.progress_bar.setVisible(True)
        self.progress_bar.setMaximum(maximum)
        self.progress_bar.setValue(value)
        self.status_label.setText(message)
    
    def on_processing_started(self):
        """Called when processing starts."""
        self.scan_button.setEnabled(False)
        self.review_button.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.status_label.setText("Processing...")
    
    def on_processing_finished(self):
        """Called when processing finishes."""
        self.scan_button.setEnabled(True)
        self.review_button.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.status_label.setText("Ready")
    
    def update_status(self):
        """Update status bar information."""
        try:
            # Update database status
            stats = self.db_manager.get_session_statistics()
            self.db_status_label.setText(
                f"DB: {stats['total_images']} images, {stats['duplicate_groups']} groups"
            )
            
            # Update session status
            if self.current_session_id:
                self.session_status_label.setText(f"Session: {self.current_session_id}")
            else:
                self.session_status_label.setText("Session: None")
                
        except Exception as e:
            logger.error(f"Error updating status: {e}")
    
    def new_session(self):
        """Create a new session."""
        session_name, ok = QInputDialog.getText(
            self, "New Session", "Enter session name:"
        )
        if ok and session_name:
            try:
                # Clear current database
                self.db_manager.clear_database()
                
                # Set session name
                self.current_session_id = session_name
                self.session_label.setText(f"Session: {session_name}")
                
                # Update config
                self.config_manager.session.current_session = session_name
                self.config_manager.save_config()
                
                logger.info(f"Created new session: {session_name}")
                QMessageBox.information(
                    self, "New Session", 
                    f"Session '{session_name}' created successfully."
                )
                
            except Exception as e:
                logger.error(f"Failed to create session: {e}")
                QMessageBox.critical(
                    self, "Error", 
                    f"Failed to create session: {e}"
                )
    
    def open_session(self):
        """Open an existing session."""
        session_file, _ = QFileDialog.getOpenFileName(
            self, "Open Session", 
            str(self.config_manager.get_session_directory()),
            "Session Files (*.json);;All Files (*)"
        )
        
        if session_file:
            try:
                # Load session configuration
                session_config = self.config_manager.load_session(session_file)
                
                # Update current session
                self.current_session_id = session_config.get('session_name', 'Unknown')
                self.session_label.setText(f"Session: {self.current_session_id}")
                
                # Update config
                self.config_manager.session.current_session = self.current_session_id
                self.config_manager.save_config()
                
                logger.info(f"Opened session: {self.current_session_id}")
                QMessageBox.information(
                    self, "Session Opened", 
                    f"Session '{self.current_session_id}' loaded successfully."
                )
                
            except Exception as e:
                logger.error(f"Failed to open session: {e}")
                QMessageBox.critical(
                    self, "Error", 
                    f"Failed to open session: {e}"
                )
    
    def save_session(self):
        """Save current session."""
        if not self.current_session_id:
            QMessageBox.warning(
                self, "No Session", 
                "No active session to save. Create a new session first."
            )
            return
        
        try:
            # Save session configuration
            session_file = self.config_manager.save_session(self.current_session_id)
            
            logger.info(f"Saved session: {self.current_session_id}")
            QMessageBox.information(
                self, "Session Saved", 
                f"Session '{self.current_session_id}' saved to:\n{session_file}"
            )
            
        except Exception as e:
            logger.error(f"Failed to save session: {e}")
            QMessageBox.critical(
                self, "Error", 
                f"Failed to save session: {e}"
            )
    
    def auto_save_session(self):
        """Auto-save current session."""
        if self.current_session_id and self.config_manager.ui.auto_save_enabled:
            try:
                self.config_manager.save_session(self.current_session_id)
                logger.debug(f"Auto-saved session: {self.current_session_id}")
            except Exception as e:
                logger.error(f"Auto-save failed: {e}")
    
    def clear_database(self):
        """Clear all data from the database."""
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
                self.current_session_id = None
                self.session_label.setText("No active session")
                logger.info("Database cleared")
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
    
    def show_about(self):
        """Show about dialog."""
        QMessageBox.about(
            self, "About Meme-Cleanup",
            "Meme-Cleanup v1.0.0\n\n"
            "A high-performance, GPU-accelerated image deduplication tool.\n\n"
            "Features:\n"
            "• CLIP-based visual similarity detection\n"
            "• BRISQUE/NIQE quality metrics\n"
            "• Session management and progress saving\n"
            "• Modern PyQt6 interface\n\n"
            "Built with Python, PyQt6, and PyTorch."
        )
    
    def closeEvent(self, event):
        """Handle application close event."""
        try:
            # Save window state
            self.config_manager.ui.window_width = self.width()
            self.config_manager.ui.window_height = self.height()
            self.config_manager.save_config()
            
            # Auto-save session if enabled
            if self.current_session_id and self.config_manager.ui.auto_save_enabled:
                self.config_manager.save_session(self.current_session_id)
            
            logger.info("Application closing")
            event.accept()
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
            event.accept() 