"""
Configuration management for Meme-Cleanup.

Handles application settings, session persistence, and user preferences.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class ProcessingConfig:
    """Configuration for image processing."""
    similarity_threshold: float = 0.8
    batch_size: int = 16
    use_gpu: bool = True
    quality_metric: str = "combined"  # "combined", "brisque", "niqe"
    parallel_processing: bool = True
    n_jobs: int = -1  # Use all cores


@dataclass
class UIConfig:
    """Configuration for user interface."""
    window_width: int = 1200
    window_height: int = 800
    window_x: int = 100
    window_y: int = 100
    dark_theme: bool = True
    auto_save_interval: int = 300  # seconds


@dataclass
class PathConfig:
    """Configuration for file paths."""
    database_path: str = ""
    log_file_path: str = ""
    output_directory: str = ""
    last_source_directories: list = None


@dataclass
class SessionConfig:
    """Configuration for session management."""
    session_name: str = ""
    created_at: str = ""
    last_modified: str = ""
    total_images: int = 0
    processed_images: int = 0
    duplicate_groups: int = 0


class ConfigManager:
    """Manages application configuration and settings."""
    
    def __init__(self, config_dir: Optional[Path] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_dir: Directory to store configuration files
        """
        if config_dir is None:
            config_dir = Path.home() / ".meme_cleanup"
        
        self.config_dir = config_dir
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        self.config_file = self.config_dir / "config.json"
        self.sessions_dir = self.config_dir / "sessions"
        self.sessions_dir.mkdir(exist_ok=True)
        
        # Default configurations
        self.processing = ProcessingConfig()
        self.ui = UIConfig()
        self.paths = PathConfig()
        self.session = SessionConfig()
        
        # Load existing configuration
        self.load_config()
        
        logger.info(f"Configuration manager initialized: {self.config_dir}")
    
    def load_config(self) -> None:
        """Load configuration from file."""
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Load processing config
                if 'processing' in data:
                    proc_data = data['processing']
                    self.processing = ProcessingConfig(
                        similarity_threshold=proc_data.get('similarity_threshold', 0.8),
                        batch_size=proc_data.get('batch_size', 16),
                        use_gpu=proc_data.get('use_gpu', True),
                        quality_metric=proc_data.get('quality_metric', 'combined'),
                        parallel_processing=proc_data.get('parallel_processing', True),
                        n_jobs=proc_data.get('n_jobs', -1)
                    )
                
                # Load UI config
                if 'ui' in data:
                    ui_data = data['ui']
                    self.ui = UIConfig(
                        window_width=ui_data.get('window_width', 1200),
                        window_height=ui_data.get('window_height', 800),
                        window_x=ui_data.get('window_x', 100),
                        window_y=ui_data.get('window_y', 100),
                        dark_theme=ui_data.get('dark_theme', True),
                        auto_save_interval=ui_data.get('auto_save_interval', 300)
                    )
                
                # Load path config
                if 'paths' in data:
                    path_data = data['paths']
                    self.paths = PathConfig(
                        database_path=path_data.get('database_path', ''),
                        log_file_path=path_data.get('log_file_path', ''),
                        output_directory=path_data.get('output_directory', ''),
                        last_source_directories=path_data.get('last_source_directories', [])
                    )
                
                logger.info("Configuration loaded successfully")
            else:
                logger.info("No configuration file found, using defaults")
                
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
    
    def save_config(self) -> None:
        """Save configuration to file."""
        try:
            config_data = {
                'processing': asdict(self.processing),
                'ui': asdict(self.ui),
                'paths': asdict(self.paths),
                'last_saved': datetime.now().isoformat()
            }
            
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)
            
            logger.info("Configuration saved successfully")
            
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
    
    def reset_config(self) -> None:
        """Reset configuration to default values."""
        self.processing = ProcessingConfig()
        self.ui = UIConfig()
        self.paths = PathConfig()
        self.session = SessionConfig()
        
        self.save_config()
        logger.info("Configuration reset to defaults")
    
    def create_session(self, name: str, source_directories: list) -> str:
        """
        Create a new session.
        
        Args:
            name: Session name
            source_directories: List of source directories
            
        Returns:
            Session ID
        """
        session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        session_data = {
            'id': session_id,
            'name': name,
            'created_at': datetime.now().isoformat(),
            'last_modified': datetime.now().isoformat(),
            'source_directories': source_directories,
            'total_images': 0,
            'processed_images': 0,
            'duplicate_groups': 0,
            'status': 'created'
        }
        
        session_file = self.sessions_dir / f"{session_id}.json"
        with open(session_file, 'w', encoding='utf-8') as f:
            json.dump(session_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Session created: {session_id}")
        return session_id
    
    def load_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Load a session by ID.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Session data or None if not found
        """
        session_file = self.sessions_dir / f"{session_id}.json"
        
        if not session_file.exists():
            logger.warning(f"Session not found: {session_id}")
            return None
        
        try:
            with open(session_file, 'r', encoding='utf-8') as f:
                session_data = json.load(f)
            
            logger.info(f"Session loaded: {session_id}")
            return session_data
            
        except Exception as e:
            logger.error(f"Failed to load session {session_id}: {e}")
            return None
    
    def save_session(self, session_id: str, session_data: Dict[str, Any]) -> None:
        """
        Save session data.
        
        Args:
            session_id: Session identifier
            session_data: Session data to save
        """
        session_file = self.sessions_dir / f"{session_id}.json"
        
        # Update last modified timestamp
        session_data['last_modified'] = datetime.now().isoformat()
        
        try:
            with open(session_file, 'w', encoding='utf-8') as f:
                json.dump(session_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Session saved: {session_id}")
            
        except Exception as e:
            logger.error(f"Failed to save session {session_id}: {e}")
    
    def list_sessions(self) -> list:
        """
        List all available sessions.
        
        Returns:
            List of session information
        """
        sessions = []
        
        for session_file in self.sessions_dir.glob("*.json"):
            try:
                with open(session_file, 'r', encoding='utf-8') as f:
                    session_data = json.load(f)
                
                sessions.append({
                    'id': session_data.get('id', session_file.stem),
                    'name': session_data.get('name', 'Unnamed Session'),
                    'created_at': session_data.get('created_at', ''),
                    'last_modified': session_data.get('last_modified', ''),
                    'status': session_data.get('status', 'unknown'),
                    'total_images': session_data.get('total_images', 0),
                    'processed_images': session_data.get('processed_images', 0)
                })
                
            except Exception as e:
                logger.error(f"Failed to read session file {session_file}: {e}")
        
        # Sort by last modified (newest first)
        sessions.sort(key=lambda x: x['last_modified'], reverse=True)
        
        return sessions
    
    def delete_session(self, session_id: str) -> bool:
        """
        Delete a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if deleted successfully
        """
        session_file = self.sessions_dir / f"{session_id}.json"
        
        if not session_file.exists():
            logger.warning(f"Session not found for deletion: {session_id}")
            return False
        
        try:
            session_file.unlink()
            logger.info(f"Session deleted: {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete session {session_id}: {e}")
            return False
    
    def get_database_path(self) -> Path:
        """Get the database file path."""
        if self.paths.database_path:
            return Path(self.paths.database_path)
        else:
            return self.config_dir / "meme_cleanup.db"
    
    def get_log_file_path(self) -> Path:
        """Get the log file path."""
        if self.paths.log_file_path:
            return Path(self.paths.log_file_path)
        else:
            return self.config_dir / "meme_cleanup.log"
    
    def update_processing_config(self, **kwargs) -> None:
        """Update processing configuration."""
        for key, value in kwargs.items():
            if hasattr(self.processing, key):
                setattr(self.processing, key, value)
        
        self.save_config()
    
    def update_ui_config(self, **kwargs) -> None:
        """Update UI configuration."""
        for key, value in kwargs.items():
            if hasattr(self.ui, key):
                setattr(self.ui, key, value)
        
        self.save_config()
    
    def update_paths_config(self, **kwargs) -> None:
        """Update paths configuration."""
        for key, value in kwargs.items():
            if hasattr(self.paths, key):
                setattr(self.paths, key, value)
        
        self.save_config() 