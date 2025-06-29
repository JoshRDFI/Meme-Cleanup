"""
Tests for configuration management functionality.
"""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import patch

from src.utils.config_manager import ConfigManager, ProcessingConfig, UIConfig, PathConfig


class TestConfigManager:
    """Test cases for ConfigManager class."""
    
    @pytest.fixture
    def temp_config_dir(self):
        """Create a temporary directory for configuration files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    @pytest.fixture
    def config_manager(self, temp_config_dir):
        """Create a ConfigManager instance with temporary directory."""
        return ConfigManager(config_dir=temp_config_dir)
    
    def test_init_default_config(self, config_manager):
        """Test initialization with default configuration."""
        assert config_manager.processing.similarity_threshold == 0.8
        assert config_manager.processing.batch_size == 16
        assert config_manager.processing.use_gpu is True
        assert config_manager.processing.quality_metric == "combined"
        assert config_manager.ui.dark_theme is True
        assert config_manager.ui.auto_save_interval == 300
    
    def test_save_and_load_config(self, config_manager):
        """Test saving and loading configuration."""
        # Modify some values
        config_manager.processing.similarity_threshold = 0.9
        config_manager.processing.batch_size = 32
        config_manager.ui.dark_theme = False
        
        # Save configuration
        config_manager.save_config()
        
        # Create new instance to test loading
        new_manager = ConfigManager(config_dir=config_manager.config_dir)
        
        # Check that values were loaded correctly
        assert new_manager.processing.similarity_threshold == 0.9
        assert new_manager.processing.batch_size == 32
        assert new_manager.ui.dark_theme is False
    
    def test_reset_config(self, config_manager):
        """Test resetting configuration to defaults."""
        # Modify some values
        config_manager.processing.similarity_threshold = 0.9
        config_manager.ui.dark_theme = False
        
        # Reset configuration
        config_manager.reset_config()
        
        # Check that values were reset to defaults
        assert config_manager.processing.similarity_threshold == 0.8
        assert config_manager.processing.batch_size == 16
        assert config_manager.ui.dark_theme is True
    
    def test_create_session(self, config_manager):
        """Test creating a new session."""
        session_id = config_manager.create_session("Test Session", ["/path1", "/path2"])
        
        assert session_id.startswith("session_")
        assert (config_manager.sessions_dir / f"{session_id}.json").exists()
        
        # Check session data
        session_data = config_manager.load_session(session_id)
        assert session_data["name"] == "Test Session"
        assert session_data["source_directories"] == ["/path1", "/path2"]
        assert session_data["status"] == "created"
    
    def test_load_session(self, config_manager):
        """Test loading an existing session."""
        # Create a session first
        session_id = config_manager.create_session("Test Session", ["/path1"])
        
        # Load the session
        session_data = config_manager.load_session(session_id)
        
        assert session_data is not None
        assert session_data["name"] == "Test Session"
        assert session_data["source_directories"] == ["/path1"]
    
    def test_load_nonexistent_session(self, config_manager):
        """Test loading a session that doesn't exist."""
        session_data = config_manager.load_session("nonexistent_session")
        assert session_data is None
    
    def test_save_session(self, config_manager):
        """Test saving session data."""
        # Create a session
        session_id = config_manager.create_session("Test Session", ["/path1"])
        
        # Modify session data
        session_data = {
            "id": session_id,
            "name": "Modified Session",
            "source_directories": ["/path1", "/path2"],
            "total_images": 100,
            "processed_images": 50,
            "duplicate_groups": 10,
            "status": "processing"
        }
        
        # Save session
        config_manager.save_session(session_id, session_data)
        
        # Load and verify
        loaded_data = config_manager.load_session(session_id)
        assert loaded_data["name"] == "Modified Session"
        assert loaded_data["total_images"] == 100
        assert loaded_data["status"] == "processing"
    
    def test_list_sessions(self, config_manager):
        """Test listing all sessions."""
        # Create multiple sessions
        config_manager.create_session("Session 1", ["/path1"])
        config_manager.create_session("Session 2", ["/path2"])
        
        # List sessions
        sessions = config_manager.list_sessions()
        
        assert len(sessions) == 2
        session_names = [s["name"] for s in sessions]
        assert "Session 1" in session_names
        assert "Session 2" in session_names
    
    def test_delete_session(self, config_manager):
        """Test deleting a session."""
        # Create a session
        session_id = config_manager.create_session("Test Session", ["/path1"])
        
        # Verify session exists
        assert config_manager.load_session(session_id) is not None
        
        # Delete session
        result = config_manager.delete_session(session_id)
        assert result is True
        
        # Verify session is deleted
        assert config_manager.load_session(session_id) is None
    
    def test_delete_nonexistent_session(self, config_manager):
        """Test deleting a session that doesn't exist."""
        result = config_manager.delete_session("nonexistent_session")
        assert result is False
    
    def test_get_database_path_default(self, config_manager):
        """Test getting default database path."""
        db_path = config_manager.get_database_path()
        assert db_path == config_manager.config_dir / "meme_cleanup.db"
    
    def test_get_database_path_custom(self, config_manager):
        """Test getting custom database path."""
        custom_path = "/custom/path/database.db"
        config_manager.paths.database_path = custom_path
        
        db_path = config_manager.get_database_path()
        assert db_path == Path(custom_path)
    
    def test_get_log_file_path_default(self, config_manager):
        """Test getting default log file path."""
        log_path = config_manager.get_log_file_path()
        assert log_path == config_manager.config_dir / "meme_cleanup.log"
    
    def test_get_log_file_path_custom(self, config_manager):
        """Test getting custom log file path."""
        custom_path = "/custom/path/log.log"
        config_manager.paths.log_file_path = custom_path
        
        log_path = config_manager.get_log_file_path()
        assert log_path == Path(custom_path)
    
    def test_update_processing_config(self, config_manager):
        """Test updating processing configuration."""
        config_manager.update_processing_config(
            similarity_threshold=0.9,
            batch_size=32,
            use_gpu=False
        )
        
        assert config_manager.processing.similarity_threshold == 0.9
        assert config_manager.processing.batch_size == 32
        assert config_manager.processing.use_gpu is False
    
    def test_update_ui_config(self, config_manager):
        """Test updating UI configuration."""
        config_manager.update_ui_config(
            dark_theme=False,
            auto_save_interval=600,
            window_width=1600
        )
        
        assert config_manager.ui.dark_theme is False
        assert config_manager.ui.auto_save_interval == 600
        assert config_manager.ui.window_width == 1600
    
    def test_update_paths_config(self, config_manager):
        """Test updating paths configuration."""
        config_manager.update_paths_config(
            database_path="/custom/db.db",
            log_file_path="/custom/log.log",
            output_directory="/custom/output"
        )
        
        assert config_manager.paths.database_path == "/custom/db.db"
        assert config_manager.paths.log_file_path == "/custom/log.log"
        assert config_manager.paths.output_directory == "/custom/output"


class TestProcessingConfig:
    """Test cases for ProcessingConfig dataclass."""
    
    def test_default_values(self):
        """Test default values for ProcessingConfig."""
        config = ProcessingConfig()
        
        assert config.similarity_threshold == 0.8
        assert config.batch_size == 16
        assert config.use_gpu is True
        assert config.quality_metric == "combined"
        assert config.parallel_processing is True
        assert config.n_jobs == -1
    
    def test_custom_values(self):
        """Test custom values for ProcessingConfig."""
        config = ProcessingConfig(
            similarity_threshold=0.9,
            batch_size=32,
            use_gpu=False,
            quality_metric="brisque",
            parallel_processing=False,
            n_jobs=4
        )
        
        assert config.similarity_threshold == 0.9
        assert config.batch_size == 32
        assert config.use_gpu is False
        assert config.quality_metric == "brisque"
        assert config.parallel_processing is False
        assert config.n_jobs == 4


class TestUIConfig:
    """Test cases for UIConfig dataclass."""
    
    def test_default_values(self):
        """Test default values for UIConfig."""
        config = UIConfig()
        
        assert config.window_width == 1200
        assert config.window_height == 800
        assert config.window_x == 100
        assert config.window_y == 100
        assert config.dark_theme is True
        assert config.auto_save_interval == 300
    
    def test_custom_values(self):
        """Test custom values for UIConfig."""
        config = UIConfig(
            window_width=1600,
            window_height=900,
            window_x=200,
            window_y=200,
            dark_theme=False,
            auto_save_interval=600
        )
        
        assert config.window_width == 1600
        assert config.window_height == 900
        assert config.window_x == 200
        assert config.window_y == 200
        assert config.dark_theme is False
        assert config.auto_save_interval == 600


class TestPathConfig:
    """Test cases for PathConfig dataclass."""
    
    def test_default_values(self):
        """Test default values for PathConfig."""
        config = PathConfig()
        
        assert config.database_path == ""
        assert config.log_file_path == ""
        assert config.output_directory == ""
        assert config.last_source_directories is None
    
    def test_custom_values(self):
        """Test custom values for PathConfig."""
        config = PathConfig(
            database_path="/custom/db.db",
            log_file_path="/custom/log.log",
            output_directory="/custom/output",
            last_source_directories=["/path1", "/path2"]
        )
        
        assert config.database_path == "/custom/db.db"
        assert config.log_file_path == "/custom/log.log"
        assert config.output_directory == "/custom/output"
        assert config.last_source_directories == ["/path1", "/path2"] 