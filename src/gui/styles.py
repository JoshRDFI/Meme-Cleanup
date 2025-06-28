"""
Dark theme styling for Meme-Cleanup.

Provides consistent dark theme styling across the application.
"""

from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QPalette, QColor


def apply_dark_theme(app: QApplication) -> None:
    """
    Apply dark theme to the application.
    
    Args:
        app: QApplication instance
    """
    # Create dark palette
    dark_palette = QPalette()
    
    # Set color roles
    dark_palette.setColor(QPalette.ColorRole.Window, QColor(30, 30, 30))
    dark_palette.setColor(QPalette.ColorRole.WindowText, QColor(245, 245, 245))
    dark_palette.setColor(QPalette.ColorRole.Base, QColor(45, 45, 45))
    dark_palette.setColor(QPalette.ColorRole.AlternateBase, QColor(60, 60, 60))
    dark_palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(45, 45, 45))
    dark_palette.setColor(QPalette.ColorRole.ToolTipText, QColor(245, 245, 245))
    dark_palette.setColor(QPalette.ColorRole.Text, QColor(245, 245, 245))
    dark_palette.setColor(QPalette.ColorRole.Button, QColor(45, 45, 45))
    dark_palette.setColor(QPalette.ColorRole.ButtonText, QColor(245, 245, 245))
    dark_palette.setColor(QPalette.ColorRole.BrightText, QColor(74, 144, 226))
    dark_palette.setColor(QPalette.ColorRole.Link, QColor(74, 144, 226))
    dark_palette.setColor(QPalette.ColorRole.Highlight, QColor(74, 144, 226))
    dark_palette.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))
    
    # Apply palette
    app.setPalette(dark_palette)
    
    # Set application style sheet
    app.setStyleSheet("""
        QWidget {
            background-color: #1E1E1E;
            color: #F5F5F5;
            font-family: 'Segoe UI', Arial, sans-serif;
            font-size: 12px;
        }
        
        QMainWindow {
            background-color: #1E1E1E;
        }
        
        QMenuBar {
            background-color: #2D2D2D;
            border-bottom: 1px solid #3D3D3D;
        }
        
        QMenuBar::item {
            background-color: transparent;
            padding: 6px 12px;
        }
        
        QMenuBar::item:selected {
            background-color: #4A90E2;
        }
        
        QMenu {
            background-color: #2D2D2D;
            border: 1px solid #3D3D3D;
        }
        
        QMenu::item {
            padding: 6px 20px;
        }
        
        QMenu::item:selected {
            background-color: #4A90E2;
        }
        
        QStatusBar {
            background-color: #2D2D2D;
            border-top: 1px solid #3D3D3D;
        }
        
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
        
        QPushButton {
            background-color: #2D2D2D;
            color: #F5F5F5;
            border: 1px solid #3D3D3D;
            padding: 8px 16px;
            border-radius: 4px;
            font-weight: bold;
        }
        
        QPushButton:hover {
            background-color: #3D3D3D;
            border-color: #4A90E2;
        }
        
        QPushButton:pressed {
            background-color: #4A90E2;
            color: white;
        }
        
        QPushButton:disabled {
            background-color: #1A1A1A;
            color: #666666;
            border-color: #2A2A2A;
        }
        
        QLineEdit {
            background-color: #2D2D2D;
            color: #F5F5F5;
            border: 1px solid #3D3D3D;
            padding: 6px 8px;
            border-radius: 4px;
        }
        
        QLineEdit:focus {
            border-color: #4A90E2;
        }
        
        QTextEdit {
            background-color: #2D2D2D;
            color: #F5F5F5;
            border: 1px solid #3D3D3D;
            border-radius: 4px;
        }
        
        QListWidget {
            background-color: #2D2D2D;
            color: #F5F5F5;
            border: 1px solid #3D3D3D;
            border-radius: 4px;
            alternate-background-color: #252525;
        }
        
        QListWidget::item {
            padding: 4px 8px;
        }
        
        QListWidget::item:selected {
            background-color: #4A90E2;
            color: white;
        }
        
        QListWidget::item:hover {
            background-color: #3D3D3D;
        }
        
        QTableWidget {
            background-color: #2D2D2D;
            color: #F5F5F5;
            border: 1px solid #3D3D3D;
            border-radius: 4px;
            alternate-background-color: #252525;
            gridline-color: #3D3D3D;
        }
        
        QTableWidget::item {
            padding: 4px 8px;
        }
        
        QTableWidget::item:selected {
            background-color: #4A90E2;
            color: white;
        }
        
        QHeaderView::section {
            background-color: #1E1E1E;
            color: #F5F5F5;
            border: 1px solid #3D3D3D;
            padding: 6px 8px;
            font-weight: bold;
        }
        
        QScrollBar:vertical {
            background-color: #2D2D2D;
            width: 12px;
            border-radius: 6px;
        }
        
        QScrollBar::handle:vertical {
            background-color: #4A90E2;
            border-radius: 6px;
            min-height: 20px;
        }
        
        QScrollBar::handle:vertical:hover {
            background-color: #357ABD;
        }
        
        QScrollBar:horizontal {
            background-color: #2D2D2D;
            height: 12px;
            border-radius: 6px;
        }
        
        QScrollBar::handle:horizontal {
            background-color: #4A90E2;
            border-radius: 6px;
            min-width: 20px;
        }
        
        QScrollBar::handle:horizontal:hover {
            background-color: #357ABD;
        }
        
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
        
        QCheckBox {
            color: #F5F5F5;
        }
        
        QCheckBox::indicator {
            width: 16px;
            height: 16px;
            border: 1px solid #3D3D3D;
            border-radius: 3px;
            background-color: #2D2D2D;
        }
        
        QCheckBox::indicator:checked {
            background-color: #4A90E2;
            border-color: #4A90E2;
        }
        
        QRadioButton {
            color: #F5F5F5;
        }
        
        QRadioButton::indicator {
            width: 16px;
            height: 16px;
            border: 1px solid #3D3D3D;
            border-radius: 8px;
            background-color: #2D2D2D;
        }
        
        QRadioButton::indicator:checked {
            background-color: #4A90E2;
            border-color: #4A90E2;
        }
        
        QComboBox {
            background-color: #2D2D2D;
            color: #F5F5F5;
            border: 1px solid #3D3D3D;
            padding: 6px 8px;
            border-radius: 4px;
        }
        
        QComboBox:focus {
            border-color: #4A90E2;
        }
        
        QComboBox::drop-down {
            border: none;
            width: 20px;
        }
        
        QComboBox::down-arrow {
            image: none;
            border-left: 5px solid transparent;
            border-right: 5px solid transparent;
            border-top: 5px solid #F5F5F5;
        }
        
        QComboBox QAbstractItemView {
            background-color: #2D2D2D;
            color: #F5F5F5;
            border: 1px solid #3D3D3D;
            selection-background-color: #4A90E2;
        }
        
        QSpinBox, QDoubleSpinBox {
            background-color: #2D2D2D;
            color: #F5F5F5;
            border: 1px solid #3D3D3D;
            padding: 6px 8px;
            border-radius: 4px;
        }
        
        QSpinBox:focus, QDoubleSpinBox:focus {
            border-color: #4A90E2;
        }
        
        QGroupBox {
            font-weight: bold;
            border: 1px solid #3D3D3D;
            border-radius: 4px;
            margin-top: 10px;
            padding-top: 10px;
        }
        
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 5px 0 5px;
        }
        
        QLabel {
            color: #F5F5F5;
        }
        
        QLabel[class="error"] {
            color: #E94E77;
        }
        
        QLabel[class="success"] {
            color: #4CAF50;
        }
        
        QLabel[class="warning"] {
            color: #FF9800;
        }
    """) 