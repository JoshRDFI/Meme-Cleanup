"""
Main entry point for the Meme-Cleanup application.

Launches the PyQt GUI and handles application initialization.
"""

import sys
import logging
from pathlib import Path
from typing import Optional

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt

from gui.main_window import MainWindow
from utils.logging_config import setup_logging
from db.database import DatabaseManager


def main() -> None:
    """Main application entry point."""
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # Initialize database
        db_manager = DatabaseManager()
        db_manager.initialize()
        
        # Create Qt application
        app = QApplication(sys.argv)
        app.setApplicationName("Meme-Cleanup")
        app.setApplicationVersion("1.0.0")
        app.setOrganizationName("Meme-Cleanup")
        
        # Set application style and properties
        app.setStyle('Fusion')  # Cross-platform consistent look
        
        # Create and show main window
        main_window = MainWindow(db_manager)
        main_window.show()
        
        logger.info("Meme-Cleanup application started successfully")
        
        # Start event loop
        sys.exit(app.exec())
        
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 