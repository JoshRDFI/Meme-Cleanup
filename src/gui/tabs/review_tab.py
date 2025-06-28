"""
Review tab for Meme-Cleanup.

Displays duplicate image groups and allows user selection of which images to keep.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QTableWidget,
    QTableWidgetItem, QHeaderView, QGroupBox, QCheckBox, QSpinBox,
    QMessageBox, QSplitter, QFrame, QScrollArea
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont, QPixmap

from db.database import DatabaseManager
from core.deduplicator import DuplicateGroup


logger = logging.getLogger(__name__)


class ReviewTab(QWidget):
    """Tab for reviewing duplicate images."""
    
    def __init__(self, db_manager: DatabaseManager):
        """
        Initialize review tab.
        
        Args:
            db_manager: Database manager instance
        """
        super().__init__()
        self.db_manager = db_manager
        self.duplicate_groups = []
        self.current_group_index = 0
        
        self.setup_ui()
        logger.info("Review tab initialized")
    
    def setup_ui(self):
        """Setup the user interface."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        
        # Title
        title_label = QLabel("Review Duplicates")
        title_font = QFont()
        title_font.setPointSize(16)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setStyleSheet("color: #4A90E2; margin-bottom: 10px;")
        layout.addWidget(title_label)
        
        # Control buttons
        control_layout = QHBoxLayout()
        
        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.clicked.connect(self.refresh_duplicates)
        control_layout.addWidget(self.refresh_button)
        
        self.auto_select_button = QPushButton("Auto-Select Best")
        self.auto_select_button.clicked.connect(self.auto_select_best)
        control_layout.addWidget(self.auto_select_button)
        
        self.clear_selections_button = QPushButton("Clear Selections")
        self.clear_selections_button.clicked.connect(self.clear_selections)
        control_layout.addWidget(self.clear_selections_button)
        
        control_layout.addStretch()
        
        # Group navigation
        self.group_label = QLabel("Group 0 of 0")
        control_layout.addWidget(self.group_label)
        
        self.prev_group_button = QPushButton("← Previous")
        self.prev_group_button.clicked.connect(self.previous_group)
        self.prev_group_button.setEnabled(False)
        control_layout.addWidget(self.prev_group_button)
        
        self.next_group_button = QPushButton("Next →")
        self.next_group_button.clicked.connect(self.next_group)
        self.next_group_button.setEnabled(False)
        control_layout.addWidget(self.next_group_button)
        
        layout.addLayout(control_layout)
        
        # Main content area
        content_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left side - duplicate groups list
        left_frame = QFrame()
        left_frame.setMaximumWidth(300)
        left_layout = QVBoxLayout(left_frame)
        
        left_layout.addWidget(QLabel("Duplicate Groups"))
        
        self.groups_table = QTableWidget()
        self.groups_table.setColumnCount(3)
        self.groups_table.setHorizontalHeaderLabels(["Group", "Images", "Selected"])
        self.groups_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.groups_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.groups_table.itemSelectionChanged.connect(self.on_group_selected)
        left_layout.addWidget(self.groups_table)
        
        content_splitter.addWidget(left_frame)
        
        # Right side - image details
        right_frame = QFrame()
        right_layout = QVBoxLayout(right_frame)
        
        right_layout.addWidget(QLabel("Group Details"))
        
        # Group info
        self.group_info_label = QLabel("No group selected")
        right_layout.addWidget(self.group_info_label)
        
        # Images table
        self.images_table = QTableWidget()
        self.images_table.setColumnCount(6)
        self.images_table.setHorizontalHeaderLabels([
            "Select", "File", "Size", "Dimensions", "Quality", "Similarity"
        ])
        self.images_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        right_layout.addWidget(self.images_table)
        
        content_splitter.addWidget(right_frame)
        
        layout.addWidget(content_splitter)
        
        # Action buttons
        action_layout = QHBoxLayout()
        
        self.export_selected_button = QPushButton("Export Selected")
        self.export_selected_button.clicked.connect(self.export_selected)
        action_layout.addWidget(self.export_selected_button)
        
        self.delete_unselected_button = QPushButton("Delete Unselected")
        self.delete_unselected_button.clicked.connect(self.delete_unselected)
        action_layout.addWidget(self.delete_unselected_button)
        
        action_layout.addStretch()
        
        layout.addLayout(action_layout)
    
    def refresh_duplicates(self):
        """Refresh the duplicate groups from database."""
        try:
            # Get duplicate groups from database
            db_groups = self.db_manager.get_duplicate_groups()
            
            # Group by group_id
            groups_dict = {}
            for row in db_groups:
                group_id = row['group_id']
                if group_id not in groups_dict:
                    groups_dict[group_id] = {
                        'group_id': group_id,
                        'group_hash': row['group_hash'],
                        'similarity_threshold': row['similarity_threshold'],
                        'images': [],
                        'selected_image_id': None
                    }
                
                image_data = {
                    'id': row['image_id'],
                    'file_path': row['file_path'],
                    'similarity_score': row['similarity_score'],
                    'is_selected': bool(row['is_selected']),
                    'brisque_score': row['brisque_score'],
                    'niqe_score': row['niqe_score'],
                    'file_size': row['file_size'],
                    'width': row['width'],
                    'height': row['height']
                }
                
                groups_dict[group_id]['images'].append(image_data)
                
                if image_data['is_selected']:
                    groups_dict[group_id]['selected_image_id'] = image_data['id']
            
            # Convert to DuplicateGroup objects
            self.duplicate_groups = []
            for group_data in groups_dict.values():
                self.duplicate_groups.append(DuplicateGroup(
                    group_id=group_data['group_id'],
                    images=group_data['images'],
                    similarity_threshold=group_data['similarity_threshold'],
                    selected_image_id=group_data['selected_image_id']
                ))
            
            # Update groups table
            self.update_groups_table()
            
            # Select first group if available
            if self.duplicate_groups:
                self.current_group_index = 0
                self.show_group(0)
            else:
                self.show_no_groups()
            
            logger.info(f"Refreshed {len(self.duplicate_groups)} duplicate groups")
            
        except Exception as e:
            logger.error(f"Failed to refresh duplicates: {e}")
            QMessageBox.critical(self, "Error", f"Failed to refresh duplicates: {e}")
    
    def update_groups_table(self):
        """Update the groups table with current data."""
        self.groups_table.setRowCount(len(self.duplicate_groups))
        
        for i, group in enumerate(self.duplicate_groups):
            # Group ID
            group_id_item = QTableWidgetItem(f"Group {group.group_id}")
            group_id_item.setData(Qt.ItemDataRole.UserRole, i)
            self.groups_table.setItem(i, 0, group_id_item)
            
            # Image count
            image_count_item = QTableWidgetItem(str(len(group.images)))
            self.groups_table.setItem(i, 1, image_count_item)
            
            # Selected count
            selected_count = sum(1 for img in group.images if img['is_selected'])
            selected_item = QTableWidgetItem(str(selected_count))
            self.groups_table.setItem(i, 2, selected_item)
        
        # Update group label
        if self.duplicate_groups:
            self.group_label.setText(f"Group {self.current_group_index + 1} of {len(self.duplicate_groups)}")
            self.prev_group_button.setEnabled(self.current_group_index > 0)
            self.next_group_button.setEnabled(self.current_group_index < len(self.duplicate_groups) - 1)
        else:
            self.group_label.setText("No groups")
            self.prev_group_button.setEnabled(False)
            self.next_group_button.setEnabled(False)
    
    def show_group(self, group_index: int):
        """Show details for a specific group."""
        if not self.duplicate_groups or group_index >= len(self.duplicate_groups):
            return
        
        group = self.duplicate_groups[group_index]
        self.current_group_index = group_index
        
        # Update group info
        self.group_info_label.setText(
            f"Group {group.group_id} - {len(group.images)} images "
            f"(Similarity threshold: {group.similarity_threshold:.2f})"
        )
        
        # Update images table
        self.images_table.setRowCount(len(group.images))
        
        for i, image in enumerate(group.images):
            # Selection checkbox
            select_checkbox = QCheckBox()
            select_checkbox.setChecked(image['is_selected'])
            select_checkbox.stateChanged.connect(
                lambda state, img=image: self.on_image_selection_changed(img, state)
            )
            self.images_table.setCellWidget(i, 0, select_checkbox)
            
            # File path
            file_path = Path(image['file_path']).name
            file_item = QTableWidgetItem(file_path)
            file_item.setToolTip(image['file_path'])
            self.images_table.setItem(i, 1, file_item)
            
            # File size
            size_mb = image['file_size'] / (1024 * 1024)
            size_item = QTableWidgetItem(f"{size_mb:.1f} MB")
            self.images_table.setItem(i, 2, size_item)
            
            # Dimensions
            dims_item = QTableWidgetItem(f"{image['width']}x{image['height']}")
            self.images_table.setItem(i, 3, dims_item)
            
            # Quality score
            quality_score = image.get('brisque_score', 'N/A')
            if quality_score != 'N/A':
                quality_score = f"{quality_score:.2f}"
            quality_item = QTableWidgetItem(str(quality_score))
            self.images_table.setItem(i, 4, quality_item)
            
            # Similarity score
            sim_item = QTableWidgetItem(f"{image['similarity_score']:.3f}")
            self.images_table.setItem(i, 5, sim_item)
        
        # Update groups table
        self.update_groups_table()
    
    def show_no_groups(self):
        """Show message when no groups are available."""
        self.group_info_label.setText("No duplicate groups found. Run a scan first.")
        self.images_table.setRowCount(0)
        self.groups_table.setRowCount(0)
        self.group_label.setText("No groups")
        self.prev_group_button.setEnabled(False)
        self.next_group_button.setEnabled(False)
    
    def on_group_selected(self):
        """Called when a group is selected in the table."""
        current_row = self.groups_table.currentRow()
        if current_row >= 0:
            item = self.groups_table.item(current_row, 0)
            group_index = item.data(Qt.ItemDataRole.UserRole)
            self.show_group(group_index)
    
    def previous_group(self):
        """Show the previous group."""
        if self.current_group_index > 0:
            self.show_group(self.current_group_index - 1)
    
    def next_group(self):
        """Show the next group."""
        if self.current_group_index < len(self.duplicate_groups) - 1:
            self.show_group(self.current_group_index + 1)
    
    def on_image_selection_changed(self, image: Dict[str, Any], state: int):
        """Called when an image selection checkbox changes."""
        selected = state == Qt.CheckState.Checked.value
        
        # Update database
        current_group = self.duplicate_groups[self.current_group_index]
        self.db_manager.mark_image_as_selected(
            current_group.group_id, image['id'], selected
        )
        
        # Update local data
        image['is_selected'] = selected
        if selected:
            current_group.selected_image_id = image['id']
        elif current_group.selected_image_id == image['id']:
            current_group.selected_image_id = None
        
        # Update groups table
        self.update_groups_table()
    
    def auto_select_best(self):
        """Automatically select the best quality image from each group."""
        try:
            for group in self.duplicate_groups:
                # Find image with best quality score
                best_image = None
                best_score = float('inf')
                
                for image in group.images:
                    quality_score = image.get('brisque_score', float('inf'))
                    if quality_score < best_score:
                        best_score = quality_score
                        best_image = image
                
                if best_image:
                    # Mark as selected in database
                    self.db_manager.mark_image_as_selected(
                        group.group_id, best_image['id'], True
                    )
                    
                    # Update local data
                    for image in group.images:
                        image['is_selected'] = (image['id'] == best_image['id'])
                    group.selected_image_id = best_image['id']
            
            # Refresh display
            self.update_groups_table()
            if self.duplicate_groups:
                self.show_group(self.current_group_index)
            
            QMessageBox.information(self, "Auto-Select", "Best quality images selected for all groups.")
            
        except Exception as e:
            logger.error(f"Failed to auto-select best images: {e}")
            QMessageBox.critical(self, "Error", f"Failed to auto-select best images: {e}")
    
    def clear_selections(self):
        """Clear all image selections."""
        try:
            for group in self.duplicate_groups:
                for image in group.images:
                    if image['is_selected']:
                        # Mark as unselected in database
                        self.db_manager.mark_image_as_selected(
                            group.group_id, image['id'], False
                        )
                        image['is_selected'] = False
                group.selected_image_id = None
            
            # Refresh display
            self.update_groups_table()
            if self.duplicate_groups:
                self.show_group(self.current_group_index)
            
            QMessageBox.information(self, "Clear Selections", "All selections cleared.")
            
        except Exception as e:
            logger.error(f"Failed to clear selections: {e}")
            QMessageBox.critical(self, "Error", f"Failed to clear selections: {e}")
    
    def export_selected(self):
        """Export selected images to a new location."""
        QMessageBox.information(self, "Export", "Export functionality not yet implemented.")
    
    def delete_unselected(self):
        """Delete unselected images from duplicate groups."""
        reply = QMessageBox.question(
            self, "Delete Unselected", 
            "This will permanently delete all unselected images from duplicate groups. Are you sure?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            QMessageBox.information(self, "Delete", "Delete functionality not yet implemented.") 