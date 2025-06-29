<<<<<<< HEAD
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
        """Setup the menu bar."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("File")
        
        # New session action
        new_session_action = QAction("New Session", self)
        new_session_action.setShortcut("Ctrl+N")
        new_session_action.triggered.connect(self.new_session)
        file_menu.addAction(new_session_action)
        
        # Open session action
        open_session_action = QAction("Open Session", self)
        open_session_action.setShortcut("Ctrl+O")
        open_session_action.triggered.connect(self.open_session)
        file_menu.addAction(open_session_action)
        
        # Save session action
        save_session_action = QAction("Save Session", self)
        save_session_action.setShortcut("Ctrl+S")
        save_session_action.triggered.connect(self.save_session)
        file_menu.addAction(save_session_action)
        
        file_menu.addSeparator()
        
        # Exit action
        exit_action = QAction("Exit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Tools menu
        tools_menu = menubar.addMenu("Tools")
        
        # Clear database action
        clear_db_action = QAction("Clear Database", self)
        clear_db_action.triggered.connect(self.clear_database)
        tools_menu.addAction(clear_db_action)
        
        # Help menu
        help_menu = menubar.addMenu("Help")
        
        # About action
        about_action = QAction("About", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
    
    def setup_status_bar(self):
        """Setup the status bar."""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        # Status label
        self.status_label = QLabel("Ready")
        self.status_bar.addWidget(self.status_label)
        
        # Session info in status bar
        self.status_bar.addPermanentWidget(QLabel("Session: None"))
    
    def quick_scan(self):
        """Quick scan action."""
        self.tab_widget.setCurrentIndex(0)  # Switch to scan tab
        self.scan_tab.quick_scan()
    
    def quick_review(self):
        """Quick review action."""
        self.tab_widget.setCurrentIndex(1)  # Switch to review tab
        self.review_tab.refresh_duplicates()
    
    def update_progress(self, value: int, maximum: int, message: str):
        """Update progress bar."""
        self.progress_bar.setVisible(True)
        self.progress_bar.setMaximum(maximum)
        self.progress_bar.setValue(value)
        self.status_label.setText(message)
    
    def on_processing_started(self):
        """Handle processing started event."""
        self.scan_button.setEnabled(False)
        self.review_button.setEnabled(False)
        self.status_label.setText("Processing started...")
    
    def on_processing_finished(self):
        """Handle processing finished event."""
        self.scan_button.setEnabled(True)
        self.review_button.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.status_label.setText("Processing completed")
        
        # Auto-save session if one is active
        if self.current_session_id:
            self.auto_save_session()
    
    def update_status(self):
        """Update status information."""
        # Update session info
        if self.current_session_id:
            session_data = self.config_manager.load_session(self.current_session_id)
            if session_data:
                self.session_label.setText(f"Session: {session_data.get('name', 'Unnamed')}")
                self.status_bar.findChild(QLabel, "").setText(f"Session: {session_data.get('name', 'Unnamed')}")
        else:
            self.session_label.setText("No active session")
            self.status_bar.findChild(QLabel, "").setText("Session: None")
    
    def new_session(self):
        """Create a new session."""
        try:
            # Get session name from user
            name, ok = QInputDialog.getText(
                self, "New Session", "Enter session name:"
            )
            
            if not ok or not name.strip():
                return
            
            # Get source directories from scan tab
            source_directories = self.scan_tab.get_source_directories()
            
            if not source_directories:
                QMessageBox.warning(
                    self, "No Directories",
                    "Please add source directories in the Scan tab first."
                )
                return
            
            # Create session
            session_id = self.config_manager.create_session(name, source_directories)
            self.current_session_id = session_id
            
            # Update UI
            self.update_status()
            
            QMessageBox.information(
                self, "Session Created",
                f"New session '{name}' created successfully."
            )
            
            logger.info(f"New session created: {session_id}")
            
        except Exception as e:
            logger.error(f"Failed to create session: {e}")
            QMessageBox.critical(self, "Error", f"Failed to create session: {e}")
    
    def open_session(self):
        """Open an existing session."""
        try:
            # Get list of available sessions
            sessions = self.config_manager.list_sessions()
            
            if not sessions:
                QMessageBox.information(
                    self, "No Sessions",
                    "No saved sessions found."
                )
                return
            
            # Create session selection dialog
            session_names = [f"{s['name']} ({s['id']})" for s in sessions]
            session_name, ok = QInputDialog.getItem(
                self, "Open Session", "Select session to open:",
                session_names, 0, False
            )
            
            if not ok:
                return
            
            # Extract session ID
            session_id = session_name.split("(")[-1].rstrip(")")
            
            # Load session
            session_data = self.config_manager.load_session(session_id)
            if session_data:
                self.current_session_id = session_id
                
                # Load source directories into scan tab
                source_dirs = session_data.get('source_directories', [])
                self.scan_tab.load_source_directories(source_dirs)
                
                # Update UI
                self.update_status()
                
                QMessageBox.information(
                    self, "Session Loaded",
                    f"Session '{session_data.get('name', 'Unnamed')}' loaded successfully."
                )
                
                logger.info(f"Session loaded: {session_id}")
            else:
                QMessageBox.warning(self, "Error", "Failed to load session.")
                
        except Exception as e:
            logger.error(f"Failed to open session: {e}")
            QMessageBox.critical(self, "Error", f"Failed to open session: {e}")
    
    def save_session(self):
        """Save current session."""
        if not self.current_session_id:
            QMessageBox.warning(
                self, "No Active Session",
                "No active session to save. Create a new session first."
            )
            return
        
        try:
            # Get current session data
            session_data = self.config_manager.load_session(self.current_session_id)
            if not session_data:
                QMessageBox.warning(self, "Error", "Failed to load current session.")
                return
            
            # Update session data with current state
            source_dirs = self.scan_tab.get_source_directories()
            session_data['source_directories'] = source_dirs
            
            # Get processing statistics from database
            total_images = self.db_manager.get_total_image_count()
            processed_images = self.db_manager.get_processed_image_count()
            duplicate_groups = self.db_manager.get_duplicate_group_count()
            
            session_data['total_images'] = total_images
            session_data['processed_images'] = processed_images
            session_data['duplicate_groups'] = duplicate_groups
            session_data['status'] = 'saved'
            
            # Save session
            self.config_manager.save_session(self.current_session_id, session_data)
            
            QMessageBox.information(
                self, "Session Saved",
                "Current session saved successfully."
            )
            
            logger.info(f"Session saved: {self.current_session_id}")
            
        except Exception as e:
            logger.error(f"Failed to save session: {e}")
            QMessageBox.critical(self, "Error", f"Failed to save session: {e}")
    
    def auto_save_session(self):
        """Auto-save current session."""
        if self.current_session_id:
            try:
                self.save_session()
                logger.debug("Auto-save completed")
            except Exception as e:
                logger.error(f"Auto-save failed: {e}")
    
    def clear_database(self):
        """Clear the database."""
        reply = QMessageBox.question(
            self, "Clear Database",
            "Are you sure you want to clear the database? This will remove all image data and cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            try:
                self.db_manager.clear_database()
                QMessageBox.information(
                    self, "Database Cleared",
                    "Database has been cleared successfully."
                )
                logger.info("Database cleared")
                
            except Exception as e:
                logger.error(f"Failed to clear database: {e}")
                QMessageBox.critical(self, "Error", f"Failed to clear database: {e}")
    
    def show_about(self):
        """Show about dialog."""
        QMessageBox.about(
            self, "About Meme-Cleanup",
            "Meme-Cleanup v1.0.0\n\n"
            "A high-performance, GPU-accelerated image deduplication tool.\n\n"
            "Features:\n"
            "• AI-powered similarity detection with CLIP\n"
            "• Quality assessment with BRISQUE/NIQE\n"
            "• GPU acceleration with CUDA\n"
            "• Modern PyQt6 interface\n\n"
            "Built with Python, PyTorch, and PyQt6."
        )
    
    def closeEvent(self, event):
        """Handle window close event."""
        # Auto-save session before closing
        if self.current_session_id:
            try:
                self.auto_save_session()
            except Exception as e:
                logger.error(f"Failed to auto-save on close: {e}")
        
        # Save window position and size
        try:
            self.config_manager.update_ui_config(
                window_width=self.width(),
                window_height=self.height(),
                window_x=self.x(),
                window_y=self.y()
            )
        except Exception as e:
            logger.error(f"Failed to save window position: {e}")
        
=======
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
    QFileDialog, QMessageBox, QSplitter, QFrame
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QAction, QFont, QPalette, QColor

from db.database import DatabaseManager
from gui.tabs.scan_tab import ScanTab
from gui.tabs.review_tab import ReviewTab
from gui.tabs.settings_tab import SettingsTab
from gui.tabs.logs_tab import LogsTab
from gui.styles import apply_dark_theme


logger = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    """Main application window."""
    
    def __init__(self, db_manager: DatabaseManager):
        """
        Initialize main window.
        
        Args:
            db_manager: Database manager instance
        """
        super().__init__()
        self.db_manager = db_manager
        
        self.setWindowTitle("Meme-Cleanup - Image Deduplication Tool")
        self.setMinimumSize(1200, 800)
        
        # Apply dark theme
        apply_dark_theme(self)
        
        # Setup UI
        self.setup_ui()
        self.setup_menu()
        self.setup_status_bar()
        
        # Setup timer for status updates
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self.update_status)
        self.status_timer.start(1000)  # Update every second
        
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
        self.scan_tab = ScanTab(self.db_manager)
        self.review_tab = ReviewTab(self.db_manager)
        self.settings_tab = SettingsTab()
        self.logs_tab = LogsTab()
        
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
        """Setup the menu bar."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("File")
        
        # New session action
        new_session_action = QAction("New Session", self)
        new_session_action.setShortcut("Ctrl+N")
        new_session_action.triggered.connect(self.new_session)
        file_menu.addAction(new_session_action)
        
        # Open session action
        open_session_action = QAction("Open Session", self)
        open_session_action.setShortcut("Ctrl+O")
        open_session_action.triggered.connect(self.open_session)
        file_menu.addAction(open_session_action)
        
        # Save session action
        save_session_action = QAction("Save Session", self)
        save_session_action.setShortcut("Ctrl+S")
        save_session_action.triggered.connect(self.save_session)
        file_menu.addAction(save_session_action)
        
        file_menu.addSeparator()
        
        # Exit action
        exit_action = QAction("Exit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Tools menu
        tools_menu = menubar.addMenu("Tools")
        
        # Clear database action
        clear_db_action = QAction("Clear Database", self)
        clear_db_action.triggered.connect(self.clear_database)
        tools_menu.addAction(clear_db_action)
        
        # Help menu
        help_menu = menubar.addMenu("Help")
        
        # About action
        about_action = QAction("About", self)
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
        
        self.image_count_label = QLabel("Images: 0")
        self.status_bar.addPermanentWidget(self.image_count_label)
        
        self.status_bar.addPermanentWidget(QLabel("|"))
        
        self.duplicate_count_label = QLabel("Duplicates: 0")
        self.status_bar.addPermanentWidget(self.duplicate_count_label)
    
    def quick_scan(self):
        """Quick scan action - switch to scan tab and start scanning."""
        self.tab_widget.setCurrentWidget(self.scan_tab)
        self.scan_tab.start_scan()
    
    def quick_review(self):
        """Quick review action - switch to review tab."""
        self.tab_widget.setCurrentWidget(self.review_tab)
        self.review_tab.refresh_duplicates()
    
    def update_progress(self, value: int, maximum: int, message: str):
        """Update progress bar."""
        self.progress_bar.setMaximum(maximum)
        self.progress_bar.setValue(value)
        self.progress_bar.setFormat(f"{message} ({value}/{maximum})")
        self.progress_bar.setVisible(True)
    
    def on_processing_started(self):
        """Called when processing starts."""
        self.scan_button.setEnabled(False)
        self.review_button.setEnabled(False)
        self.status_label.setText("Processing...")
    
    def on_processing_finished(self):
        """Called when processing finishes."""
        self.scan_button.setEnabled(True)
        self.review_button.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.status_label.setText("Ready")
        self.update_status()
    
    def update_status(self):
        """Update status bar information."""
        try:
            # Get image count
            all_images = self.db_manager.get_all_images()
            self.image_count_label.setText(f"Images: {len(all_images)}")
            
            # Get duplicate count
            duplicate_groups = self.db_manager.get_duplicate_groups()
            total_duplicates = sum(len([row for row in duplicate_groups if row['group_id'] == group_id]) 
                                 for group_id in set(row['group_id'] for row in duplicate_groups))
            self.duplicate_count_label.setText(f"Duplicates: {total_duplicates}")
            
        except Exception as e:
            logger.error(f"Failed to update status: {e}")
    
    def new_session(self):
        """Start a new session."""
        reply = QMessageBox.question(
            self, "New Session", 
            "This will clear all current data. Are you sure?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            try:
                self.db_manager.clear_database()
                self.update_status()
                QMessageBox.information(self, "New Session", "Session cleared successfully.")
            except Exception as e:
                logger.error(f"Failed to clear database: {e}")
                QMessageBox.critical(self, "Error", f"Failed to clear database: {e}")
    
    def open_session(self):
        """Open a saved session."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Session", "", "Database Files (*.db);;All Files (*)"
        )
        
        if file_path:
            try:
                # TODO: Implement session loading
                QMessageBox.information(self, "Open Session", "Session loading not yet implemented.")
            except Exception as e:
                logger.error(f"Failed to open session: {e}")
                QMessageBox.critical(self, "Error", f"Failed to open session: {e}")
    
    def save_session(self):
        """Save current session."""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Session", "", "Database Files (*.db);;All Files (*)"
        )
        
        if file_path:
            try:
                # TODO: Implement session saving
                QMessageBox.information(self, "Save Session", "Session saving not yet implemented.")
            except Exception as e:
                logger.error(f"Failed to save session: {e}")
                QMessageBox.critical(self, "Error", f"Failed to save session: {e}")
    
    def clear_database(self):
        """Clear the database."""
        reply = QMessageBox.question(
            self, "Clear Database", 
            "This will permanently delete all data. Are you sure?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            try:
                self.db_manager.clear_database()
                self.update_status()
                QMessageBox.information(self, "Clear Database", "Database cleared successfully.")
            except Exception as e:
                logger.error(f"Failed to clear database: {e}")
                QMessageBox.critical(self, "Error", f"Failed to clear database: {e}")
    
    def show_about(self):
        """Show about dialog."""
        QMessageBox.about(
            self, "About Meme-Cleanup",
            "Meme-Cleanup v1.0.0\n\n"
            "A high-performance, GPU-accelerated image deduplication tool.\n\n"
            "Features:\n"
            "• CLIP-based visual similarity detection\n"
            "• BRISQUE/NIQE quality assessment\n"
            "• Batch processing with progress tracking\n"
            "• Interactive duplicate review\n"
            "• Cross-platform PyQt6 interface\n\n"
            "Built with Python, PyQt6, PyTorch, and CLIP."
        )
    
    def closeEvent(self, event):
        """Handle application close event."""
        # Stop any running processes
        if hasattr(self.scan_tab, 'stop_processing'):
            self.scan_tab.stop_processing()
        
        # Save any unsaved progress
        # TODO: Implement auto-save functionality
        
>>>>>>> 88a757b8ecc4ae4d819ae02a34c56b2a1ebc3714
        event.accept() 