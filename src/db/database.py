"""
Database manager for Meme-Cleanup.

Handles SQLite database initialization, schema management, and connection pooling.
"""

import logging
import sqlite3
from pathlib import Path
from typing import Optional, List, Dict, Any
from contextlib import contextmanager


logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages SQLite database operations for Meme-Cleanup."""
    
    def __init__(self, db_path: Optional[Path] = None):
        """
        Initialize database manager.
        
        Args:
            db_path: Path to SQLite database file (default: creates in user data directory)
        """
        if db_path is None:
            # Use user data directory
            user_data_dir = Path.home() / ".meme_cleanup"
            user_data_dir.mkdir(exist_ok=True)
            db_path = user_data_dir / "meme_cleanup.db"
        
        self.db_path = db_path
        self._connection: Optional[sqlite3.Connection] = None
        
    def initialize(self) -> None:
        """Initialize database and create tables if they don't exist."""
        try:
            with self._get_connection() as conn:
                self._create_tables(conn)
            logger.info(f"Database initialized at {self.db_path}")
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise
    
    @contextmanager
    def _get_connection(self):
        """Get database connection with proper error handling."""
        conn = None
        try:
            conn = sqlite3.connect(str(self.db_path))
            conn.row_factory = sqlite3.Row  # Enable dict-like access
            yield conn
        except Exception as e:
            logger.error(f"Database connection error: {e}")
            raise
        finally:
            if conn:
                conn.close()
    
    def _create_tables(self, conn: sqlite3.Connection) -> None:
        """Create database tables if they don't exist."""
        # Images table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT UNIQUE NOT NULL,
                file_size INTEGER NOT NULL,
                file_modified REAL NOT NULL,
                width INTEGER,
                height INTEGER,
                format TEXT,
                mode TEXT,
                dpi_x REAL,
                dpi_y REAL,
                perceptual_hash TEXT,
                clip_embedding BLOB,
                brisque_score REAL,
                niqe_score REAL,
                has_alpha BOOLEAN DEFAULT 0,
                is_animated BOOLEAN DEFAULT 0,
                frame_count INTEGER DEFAULT 1,
                metadata_richness INTEGER DEFAULT 0,
                exif_data TEXT,
                processed_at REAL,
                created_at REAL DEFAULT (strftime('%s', 'now'))
            )
        """)
        
        # Duplicate groups table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS duplicate_groups (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                group_hash TEXT UNIQUE NOT NULL,
                similarity_threshold REAL NOT NULL,
                created_at REAL DEFAULT (strftime('%s', 'now'))
            )
        """)
        
        # Duplicate images table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS duplicate_images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                group_id INTEGER NOT NULL,
                image_id INTEGER NOT NULL,
                similarity_score REAL NOT NULL,
                is_selected BOOLEAN DEFAULT 0,
                created_at REAL DEFAULT (strftime('%s', 'now')),
                FOREIGN KEY (group_id) REFERENCES duplicate_groups (id),
                FOREIGN KEY (image_id) REFERENCES images (id),
                UNIQUE(group_id, image_id)
            )
        """)
        
        # Processing sessions table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS processing_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_name TEXT NOT NULL,
                source_directories TEXT NOT NULL,
                output_directory TEXT NOT NULL,
                status TEXT DEFAULT 'pending',
                total_images INTEGER DEFAULT 0,
                processed_images INTEGER DEFAULT 0,
                created_at REAL DEFAULT (strftime('%s', 'now')),
                updated_at REAL DEFAULT (strftime('%s', 'now'))
            )
        """)
        
        # Create scan_results table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS scan_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                directories_scanned INTEGER NOT NULL,
                total_images_processed INTEGER NOT NULL,
                total_images_found INTEGER NOT NULL,
                corrupted_files_skipped INTEGER NOT NULL,
                skipped_files_count INTEGER NOT NULL,
                scan_duration REAL NOT NULL,
                scan_completed_at REAL NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create indexes for better performance
        conn.execute("CREATE INDEX IF NOT EXISTS idx_images_file_path ON images(file_path)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_images_perceptual_hash ON images(perceptual_hash)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_duplicate_images_group_id ON duplicate_images(group_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_duplicate_images_image_id ON duplicate_images(image_id)")
        
        conn.commit()
    
    def insert_image(self, image_data: Dict[str, Any]) -> int:
        """
        Insert or update image record.
        
        Args:
            image_data: Dictionary containing image metadata
            
        Returns:
            Image ID
        """
        with self._get_connection() as conn:
            # Filter out EXIF data and other non-serializable fields
            safe_image_data = {
                'file_path': image_data['file_path'],
                'file_size': image_data['file_size'],
                'file_modified': image_data['file_modified'],
                'width': image_data.get('width'),
                'height': image_data.get('height'),
                'format': image_data.get('format'),
                'mode': image_data.get('mode'),
                'dpi_x': image_data.get('dpi', (None, None))[0] if image_data.get('dpi') else None,
                'dpi_y': image_data.get('dpi', (None, None))[1] if image_data.get('dpi') else None,
                'perceptual_hash': image_data.get('perceptual_hash'),
                'processed_at': image_data.get('processed_at')
            }
            
            cursor = conn.execute("""
                INSERT OR REPLACE INTO images (
                    file_path, file_size, file_modified, width, height,
                    format, mode, dpi_x, dpi_y, perceptual_hash,
                    processed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                safe_image_data['file_path'],
                safe_image_data['file_size'],
                safe_image_data['file_modified'],
                safe_image_data['width'],
                safe_image_data['height'],
                safe_image_data['format'],
                safe_image_data['mode'],
                safe_image_data['dpi_x'],
                safe_image_data['dpi_y'],
                safe_image_data['perceptual_hash'],
                safe_image_data['processed_at']
            ))
            
            conn.commit()
            return cursor.lastrowid
    
    def update_image_embeddings(self, image_id: int, clip_embedding: bytes, 
                               brisque_score: float, niqe_score: float) -> None:
        """
        Update image with CLIP embeddings and quality scores.
        
        Args:
            image_id: ID of the image to update
            clip_embedding: CLIP embedding as bytes
            brisque_score: BRISQUE quality score
            niqe_score: NIQE quality score
        """
        with self._get_connection() as conn:
            conn.execute("""
                UPDATE images 
                SET clip_embedding = ?, brisque_score = ?, niqe_score = ?, 
                    processed_at = strftime('%s', 'now')
                WHERE id = ?
            """, (clip_embedding, brisque_score, niqe_score, image_id))
            conn.commit()
    
    def get_image_by_path(self, file_path: str) -> Optional[Dict[str, Any]]:
        """
        Get image record by file path.
        
        Args:
            file_path: Path to the image file
            
        Returns:
            Image record as dictionary or None if not found
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM images WHERE file_path = ?", 
                (file_path,)
            )
            row = cursor.fetchone()
            return dict(row) if row else None
    
    def get_unprocessed_images(self) -> List[Dict[str, Any]]:
        """
        Get all images that haven't been processed for embeddings.
        
        Returns:
            List of unprocessed image records
        """
        with self._get_connection() as conn:
            cursor = conn.execute("""
                SELECT * FROM images 
                WHERE clip_embedding IS NULL
                ORDER BY created_at ASC
            """)
            return [dict(row) for row in cursor.fetchall()]
    
    def get_all_images(self) -> List[Dict[str, Any]]:
        """
        Get all images in the database.
        
        Returns:
            List of all image records
        """
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT * FROM images ORDER BY created_at DESC")
            return [dict(row) for row in cursor.fetchall()]
    
    def create_duplicate_group(self, group_hash: str, similarity_threshold: float) -> int:
        """
        Create a new duplicate group.
        
        Args:
            group_hash: Unique hash for the group
            similarity_threshold: Threshold used for grouping
            
        Returns:
            Group ID
        """
        with self._get_connection() as conn:
            cursor = conn.execute("""
                INSERT INTO duplicate_groups (group_hash, similarity_threshold)
                VALUES (?, ?)
            """, (group_hash, similarity_threshold))
            conn.commit()
            return cursor.lastrowid
    
    def add_image_to_duplicate_group(self, group_id: int, image_id: int, 
                                   similarity_score: float) -> None:
        """
        Add an image to a duplicate group.
        
        Args:
            group_id: ID of the duplicate group
            image_id: ID of the image to add
            similarity_score: Similarity score for this image
        """
        with self._get_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO duplicate_images 
                (group_id, image_id, similarity_score)
                VALUES (?, ?, ?)
            """, (group_id, image_id, similarity_score))
            conn.commit()
    
    def get_duplicate_groups(self) -> List[Dict[str, Any]]:
        """
        Get all duplicate groups with their images.
        
        Returns:
            List of duplicate groups with nested image data
        """
        with self._get_connection() as conn:
            # Get all groups
            cursor = conn.execute("""
                SELECT * FROM duplicate_groups 
                ORDER BY created_at DESC
            """)
            groups = [dict(row) for row in cursor.fetchall()]
            
            # Get images for each group
            for group in groups:
                cursor = conn.execute("""
                    SELECT i.*, di.similarity_score, di.is_selected
                    FROM images i
                    JOIN duplicate_images di ON i.id = di.image_id
                    WHERE di.group_id = ?
                    ORDER BY di.similarity_score DESC
                """, (group['id'],))
                group['images'] = [dict(row) for row in cursor.fetchall()]
            
            return groups
    
    def mark_image_as_selected(self, group_id: int, image_id: int, selected: bool) -> None:
        """
        Mark an image as selected or deselected in a duplicate group.
        
        Args:
            group_id: ID of the duplicate group
            image_id: ID of the image to mark
            selected: Whether the image should be selected
        """
        with self._get_connection() as conn:
            conn.execute("""
                UPDATE duplicate_images 
                SET is_selected = ?
                WHERE group_id = ? AND image_id = ?
            """, (selected, group_id, image_id))
            conn.commit()
    
    def clear_database(self) -> None:
        """Clear all data from the database."""
        with self._get_connection() as conn:
            conn.execute("DELETE FROM duplicate_images")
            conn.execute("DELETE FROM duplicate_groups")
            conn.execute("DELETE FROM images")
            conn.execute("DELETE FROM scan_results")
            conn.commit()
    
    def save_scan_results(self, scan_results: Dict[str, Any]) -> None:
        """
        Save scan results to database.
        
        Args:
            scan_results: Dictionary containing scan statistics
        """
        with self._get_connection() as conn:
            conn.execute("""
                INSERT INTO scan_results (
                    directories_scanned, total_images_processed, total_images_found,
                    corrupted_files_skipped, skipped_files_count, scan_duration,
                    scan_completed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                scan_results['directories_scanned'],
                scan_results['total_images_processed'],
                scan_results['total_images_found'],
                scan_results['corrupted_files_skipped'],
                scan_results['skipped_files_count'],
                scan_results['scan_duration'],
                scan_results['scan_completed_at']
            ))
            conn.commit()
    
    def get_selected_images(self) -> List[Dict[str, Any]]:
        """
        Get all images that have been selected in duplicate groups.
        
        Returns:
            List of selected image records
        """
        with self._get_connection() as conn:
            cursor = conn.execute("""
                SELECT i.*, di.similarity_score, di.is_selected
                FROM images i
                JOIN duplicate_images di ON i.id = di.image_id
                WHERE di.is_selected = 1
                ORDER BY di.similarity_score DESC
            """)
            return [dict(row) for row in cursor.fetchall()]
    
    def get_all_processed_images(self) -> List[Dict[str, Any]]:
        """
        Get all images that have been processed (have embeddings).
        
        Returns:
            List of processed image records
        """
        with self._get_connection() as conn:
            cursor = conn.execute("""
                SELECT * FROM images 
                WHERE clip_embedding IS NOT NULL
                ORDER BY processed_at DESC
            """)
            return [dict(row) for row in cursor.fetchall()]
    
    def get_total_image_count(self) -> int:
        """
        Get total number of images in database.
        
        Returns:
            Total count of images
        """
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT COUNT(*) as count FROM images")
            return cursor.fetchone()['count']
    
    def get_processed_image_count(self) -> int:
        """
        Get number of processed images.
        
        Returns:
            Count of processed images
        """
        with self._get_connection() as conn:
            cursor = conn.execute("""
                SELECT COUNT(*) as count FROM images 
                WHERE clip_embedding IS NOT NULL
            """)
            return cursor.fetchone()['count']
    
    def get_duplicate_group_count(self) -> int:
        """
        Get number of duplicate groups.
        
        Returns:
            Count of duplicate groups
        """
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT COUNT(*) as count FROM duplicate_groups")
            return cursor.fetchone()['count']
    
    def get_session_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive session statistics.
        
        Returns:
            Dictionary with session statistics
        """
        with self._get_connection() as conn:
            # Get basic counts
            total_images = conn.execute("SELECT COUNT(*) as count FROM images").fetchone()['count']
            processed_images = conn.execute("""
                SELECT COUNT(*) as count FROM images WHERE clip_embedding IS NOT NULL
            """).fetchone()['count']
            duplicate_groups = conn.execute("SELECT COUNT(*) as count FROM duplicate_groups").fetchone()['count']
            
            # Get total images in duplicate groups
            duplicate_images = conn.execute("SELECT COUNT(*) as count FROM duplicate_images").fetchone()['count']
            
            # Get selected images count
            selected_images = conn.execute("""
                SELECT COUNT(*) as count FROM duplicate_images WHERE is_selected = 1
            """).fetchone()['count']
            
            # Get latest scan results
            latest_scan = conn.execute("""
                SELECT * FROM scan_results ORDER BY created_at DESC LIMIT 1
            """).fetchone()
            
            return {
                'total_images': total_images,
                'processed_images': processed_images,
                'duplicate_groups': duplicate_groups,
                'duplicate_images': duplicate_images,
                'selected_images': selected_images,
                'latest_scan': dict(latest_scan) if latest_scan else None
            }
