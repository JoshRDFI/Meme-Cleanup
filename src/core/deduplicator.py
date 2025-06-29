<<<<<<< HEAD
"""
Main deduplication engine for Meme-Cleanup.

Orchestrates the entire deduplication process including scanning, embedding generation,
duplicate detection, and quality assessment.
"""

import logging
import time
import hashlib
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from joblib import Parallel, delayed

from db.database import DatabaseManager
from core.clip_processor import CLIPEmbeddingProcessor
from core.quality_metrics import QualityMetricsProcessor
from utils.image_utils import get_valid_image_files, extract_image_metadata, calculate_image_hash, load_image, is_valid_image_file, get_all_media_files, is_supported_image, is_video_file


logger = logging.getLogger(__name__)


@dataclass
class DeduplicationConfig:
    """Configuration for deduplication process."""
    similarity_threshold: float = 0.8
    batch_size: int = 8
    max_workers: int = 4
    use_gpu: bool = True
    quality_metric: str = "brisque"  # Default to BRISQUE for artifact detection
    save_progress: bool = True
    parallel_processing: bool = True
    n_jobs: int = -1  # Use all available cores


@dataclass
class DuplicateGroup:
    """Represents a group of duplicate images."""
    group_id: int
    images: List[Dict[str, Any]]
    similarity_threshold: float
    selected_image_id: Optional[int] = None


class Deduplicator:
    """Main deduplication engine."""
    
    def __init__(self, db_manager: DatabaseManager, config: DeduplicationConfig):
        """
        Initialize deduplicator.
        
        Args:
            db_manager: Database manager instance
            config: Deduplication configuration
        """
        self.db_manager = db_manager
        self.config = config
        
        # Initialize processors
        device = "cuda" if config.use_gpu else "cpu"
        self.clip_processor = CLIPEmbeddingProcessor(device=device)
        self.quality_processor = QualityMetricsProcessor()
        
        logger.info(f"Deduplicator initialized with config: {config}")
    
    def scan_directories(self, directories: List[Path]) -> List[Path]:
        """
        Scan directories for image files and store metadata using parallel processing.
        
        Args:
            directories: List of directories to scan
            
        Returns:
            List of discovered image file paths
        """
        logger.info(f"Scanning {len(directories)} directories for images")
        
        all_image_files = []
        processed_count = 0
        corrupted_files_count = 0
        skipped_files_count = 0
        start_time = time.time()
        
        for directory in directories:
            if not directory.exists():
                logger.warning(f"Directory does not exist: {directory}")
                continue
            
            logger.info(f"Scanning directory: {directory}")
            
            # Get all files first to see what we're working with
            all_files = list(directory.rglob('*'))
            image_files = get_valid_image_files(directory, recursive=True)
            
            logger.info(f"Directory {directory}:")
            logger.info(f"  Total files found: {len(all_files)}")
            logger.info(f"  Valid image files: {len(image_files)}")
            
            # Count corrupted files that were filtered out
            _, _, corrupted = get_all_media_files(directory, recursive=True)
            corrupted_files_count += len(corrupted)
            
            # Count skipped files (non-image, non-video)
            skipped_files = [f for f in all_files if f.is_file() and 
                           not is_supported_image(f) and 
                           not is_video_file(f)]
            skipped_files_count += len(skipped_files)
            
            logger.info(f"  Corrupted files: {len(corrupted)}")
            logger.info(f"  Skipped files: {len(skipped_files)}")
            
            if len(skipped_files) > 0:
                logger.info(f"  Sample skipped files:")
                for f in skipped_files[:10]:  # Show first 10
                    logger.info(f"    - {f.name}")
                if len(skipped_files) > 10:
                    logger.info(f"    ... and {len(skipped_files) - 10} more")
            
            # Process images in parallel if enabled
            if self.config.parallel_processing and len(image_files) > 10:
                logger.info(f"Using parallel processing for {len(image_files)} images")
                
                def process_single_image(image_path):
                    try:
                        # Double-check file validity before processing
                        if not is_valid_image_file(image_path):
                            logger.warning(f"Skipping corrupted file: {image_path}")
                            return None
                        
                        # Extract metadata
                        metadata = extract_image_metadata(image_path)
                        
                        # Calculate perceptual hash
                        image = load_image(image_path)
                        if image is not None:
                            metadata['perceptual_hash'] = calculate_image_hash(image)
                            metadata['processed_at'] = time.time()
                            return metadata
                    except Exception as e:
                        logger.error(f"Failed to process {image_path}: {e}")
                    return None
                
                # Process in parallel
                results = Parallel(n_jobs=self.config.n_jobs, verbose=1)(
                    delayed(process_single_image)(image_path) for image_path in image_files
                )
                
                # Store results in database
                successful_count = 0
                failed_count = 0
                for metadata in results:
                    if metadata:
                        try:
                            self.db_manager.insert_image(metadata)
                            successful_count += 1
                        except Exception as e:
                            logger.error(f"Failed to insert {metadata.get('file_path', 'unknown')}: {e}")
                            failed_count += 1
                    else:
                        failed_count += 1
                
                processed_count += successful_count
                logger.info(f"  Parallel processing results: {successful_count} successful, {failed_count} failed")
                
            else:
                # Sequential processing for small batches
                successful_count = 0
                failed_count = 0
                for image_path in image_files:
                    try:
                        # Extract metadata
                        metadata = extract_image_metadata(image_path)
                        
                        # Calculate perceptual hash
                        image = load_image(image_path)
                        if image is not None:
                            metadata['perceptual_hash'] = calculate_image_hash(image)
                            metadata['processed_at'] = time.time()
                            
                            # Store in database
                            self.db_manager.insert_image(metadata)
                            successful_count += 1
                        else:
                            failed_count += 1
                    
                    except Exception as e:
                        logger.error(f"Failed to process {image_path}: {e}")
                        failed_count += 1
                
                processed_count += successful_count
                logger.info(f"  Sequential processing results: {successful_count} successful, {failed_count} failed")
            
            all_image_files.extend(image_files)
        
        # Store scan results
        scan_results = {
            'directories_scanned': len(directories),
            'total_images_processed': processed_count,
            'total_images_found': len(all_image_files),
            'corrupted_files_skipped': corrupted_files_count,
            'skipped_files_count': skipped_files_count,
            'scan_duration': time.time() - start_time,
            'scan_completed_at': time.time()
        }
        
        self.db_manager.save_scan_results(scan_results)
        
        logger.info(f"Scan completed:")
        logger.info(f"  Total images found: {len(all_image_files)}")
        logger.info(f"  Successfully processed: {processed_count}")
        logger.info(f"  Corrupted files skipped: {corrupted_files_count}")
        logger.info(f"  Other files skipped: {skipped_files_count}")
        logger.info(f"  Scan duration: {scan_results['scan_duration']:.2f} seconds")
        
        return all_image_files
    
    def generate_embeddings(self, image_paths: Optional[List[Path]] = None) -> None:
        """
        Generate CLIP embeddings for images.
        
        Args:
            image_paths: List of image paths to process (None for all unprocessed)
        """
        if image_paths is None:
            # Get unprocessed images from database
            unprocessed_images = self.db_manager.get_unprocessed_images()
            image_paths = [Path(img['file_path']) for img in unprocessed_images]
        
        logger.info(f"Generating embeddings for {len(image_paths)} images")
        
        # Process in batches
        for i in range(0, len(image_paths), self.config.batch_size):
            batch_paths = image_paths[i:i + self.config.batch_size]
            
            try:
                # Generate embeddings
                embeddings = self.clip_processor.get_embeddings_batch(
                    batch_paths, self.config.batch_size
                )
                
                # Update database with embeddings and quality scores
                for j, (image_path, embedding) in enumerate(zip(batch_paths, embeddings)):
                    if embedding is not None:
                        # Get image record from database
                        image_record = self.db_manager.get_image_by_path(str(image_path))
                        if image_record:
                            # Calculate quality scores
                            quality_metrics = self.quality_processor.calculate_all_metrics(image_path)
                            
                            if quality_metrics:
                                brisque_score, niqe_score, combined_score = quality_metrics
                                
                                # Update database
                                self.db_manager.update_image_embeddings(
                                    image_record['id'],
                                    embedding.tobytes(),
                                    brisque_score,
                                    niqe_score
                                )
                
                logger.info(f"Processed batch {i//self.config.batch_size + 1}/{(len(image_paths) + self.config.batch_size - 1)//self.config.batch_size}")
                
            except Exception as e:
                logger.error(f"Failed to process batch {i//self.config.batch_size + 1}: {e}")
    
    def find_duplicates(self) -> List[DuplicateGroup]:
        """
        Find duplicate images using CLIP embeddings.
        
        Returns:
            List of duplicate groups
        """
        logger.info("Finding duplicate images")
        
        # Get all processed images
        all_images = self.db_manager.get_all_images()
        processed_images = [img for img in all_images if img['clip_embedding'] is not None]
        
        logger.info(f"Processing {len(processed_images)} images for duplicates")
        
        if len(processed_images) < 2:
            logger.warning("Not enough processed images to find duplicates (need at least 2)")
            return []
        
        # Convert embeddings back to numpy arrays
        embeddings = []
        image_ids = []
        image_paths = []
        
        for img in processed_images:
            try:
                embedding = np.frombuffer(img['clip_embedding'], dtype=np.float32)
                embeddings.append(embedding)
                image_ids.append(img['id'])
                image_paths.append(img['file_path'])
                logger.debug(f"Loaded embedding for: {img['file_path']}")
            except Exception as e:
                logger.error(f"Failed to load embedding for image {img['id']}: {e}")
        
        logger.info(f"Successfully loaded {len(embeddings)} embeddings")
        
        # Find duplicate groups
        duplicate_groups = []
        processed_indices = set()
        
        for i, embedding in enumerate(embeddings):
            if i in processed_indices:
                continue
            
            logger.debug(f"Checking for duplicates of: {image_paths[i]}")
            
            # Find similar images
            similar_indices = self.clip_processor.find_similar_images(
                embedding, embeddings, self.config.similarity_threshold
            )
            
            logger.debug(f"Found {len(similar_indices)} similar images for {image_paths[i]}")
            
            # Filter out already processed images
            similar_indices = [(idx, score) for idx, score in similar_indices 
                             if idx not in processed_indices]
            
            if len(similar_indices) > 1:  # More than just the image itself
                logger.info(f"Creating duplicate group with {len(similar_indices)} images")
                
                # Create duplicate group
                group_images = []
                for idx, score in similar_indices:
                    image_record = next(img for img in processed_images if img['id'] == image_ids[idx])
                    group_images.append({
                        'id': image_record['id'],
                        'file_path': image_record['file_path'],
                        'similarity_score': score,
                        'brisque_score': image_record['brisque_score'],
                        'niqe_score': image_record['niqe_score'],
                        'file_size': image_record['file_size'],
                        'width': image_record['width'],
                        'height': image_record['height']
                    })
                    processed_indices.add(idx)
                    logger.info(f"  Added to group: {image_record['file_path']} (score: {score:.3f})")
                
                # Sort by quality score (best first)
                group_images.sort(key=lambda x: x.get('brisque_score', float('inf')))
                
                # Create group in database
                group_hash = hashlib.md5(f"group_{len(duplicate_groups)}".encode()).hexdigest()
                group_id = self.db_manager.create_duplicate_group(group_hash, self.config.similarity_threshold)
                
                # Add images to group
                for img in group_images:
                    self.db_manager.add_image_to_duplicate_group(
                        group_id, img['id'], img['similarity_score']
                    )
                
                duplicate_groups.append(DuplicateGroup(
                    group_id=group_id,
                    images=group_images,
                    similarity_threshold=self.config.similarity_threshold
                ))
            else:
                logger.debug(f"No duplicates found for {image_paths[i]}")
        
        logger.info(f"Found {len(duplicate_groups)} duplicate groups")
        
        # Log some statistics about the embeddings
        if embeddings:
            # Calculate some basic statistics
            all_embeddings = np.array(embeddings)
            mean_similarity = np.mean(all_embeddings)
            std_similarity = np.std(all_embeddings)
            logger.info(f"Embedding statistics - Mean: {mean_similarity:.4f}, Std: {std_similarity:.4f}")
            logger.info(f"Similarity threshold: {self.config.similarity_threshold}")
        
        return duplicate_groups
    
    def select_best_images(self, duplicate_groups: List[DuplicateGroup]) -> None:
        """
        Automatically select the best quality image from each duplicate group.
        
        Args:
            duplicate_groups: List of duplicate groups
        """
        logger.info("Selecting best quality images from duplicate groups")
        
        for group in duplicate_groups:
            # Find image with best quality score
            best_image = None
            best_score = float('inf')
            
            for img in group.images:
                # Use combined quality score if available, otherwise BRISQUE
                quality_score = img.get('brisque_score', float('inf'))
                if img.get('niqe_score') is not None:
                    # Calculate combined score (simplified)
                    combined_score = (quality_score + img['niqe_score']) / 2
                    quality_score = combined_score
                
                if quality_score < best_score:
                    best_score = quality_score
                    best_image = img
            
            if best_image:
                # Mark as selected in database
                self.db_manager.mark_image_as_selected(
                    group.group_id, best_image['id'], True
                )
                group.selected_image_id = best_image['id']
                
                logger.info(f"Selected {best_image['file_path']} for group {group.group_id}")
    
    def run_full_deduplication(self, source_directories: List[Path]) -> List[DuplicateGroup]:
        """
        Run the complete deduplication process.
        
        Args:
            source_directories: List of source directories to process
            
        Returns:
            List of duplicate groups found
        """
        logger.info("Starting full deduplication process")
        
        try:
            # Step 1: Scan directories
            image_files = self.scan_directories(source_directories)
            
            # Step 2: Generate embeddings and quality scores
            self.generate_embeddings()
            
            # Step 3: Find duplicates
            duplicate_groups = self.find_duplicates()
            
            # Step 4: Select best images
            self.select_best_images(duplicate_groups)
            
            logger.info("Deduplication process completed successfully")
            return duplicate_groups
            
        except Exception as e:
            logger.error(f"Deduplication process failed: {e}")
            raise
    
    def get_duplicate_groups(self) -> List[DuplicateGroup]:
        """
        Get all duplicate groups from database.
        
        Returns:
            List of duplicate groups
        """
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
        duplicate_groups = []
        for group_data in groups_dict.values():
            duplicate_groups.append(DuplicateGroup(
                group_id=group_data['group_id'],
                images=group_data['images'],
                similarity_threshold=group_data['similarity_threshold'],
                selected_image_id=group_data['selected_image_id']
            ))
        
        return duplicate_groups
    
    def get_scan_results(self) -> Dict[str, Any]:
        """Get the latest scan results from the database."""
        try:
            cursor = self.db_manager._get_connection().cursor()
            cursor.execute("""
                SELECT * FROM scan_results 
                ORDER BY scan_completed_at DESC 
                LIMIT 1
            """)
            result = cursor.fetchone()
            
            if result:
                return {
                    'directories_scanned': result[1],
                    'total_images_processed': result[2],
                    'total_images_found': result[3],
                    'corrupted_files_skipped': result[4],
                    'skipped_files_count': result[5],
                    'scan_duration': result[6],
                    'scan_completed_at': result[7]
                }
            else:
                return {}
        except Exception as e:
            logger.error(f"Failed to get scan results: {e}")
            return {}
    
    def consolidate_files(self, output_directory: Path, preserve_structure: bool = True, 
                         copy_mode: bool = True) -> Dict[str, Any]:
        """
        Consolidate selected images into a single output directory.
        
        Args:
            output_directory: Target directory for consolidated files
            preserve_structure: Whether to preserve subdirectory structure
            copy_mode: True to copy files, False to move files
            
        Returns:
            Dictionary with consolidation results
        """
        logger.info(f"Starting file consolidation to {output_directory}")
        
        # Create output directory if it doesn't exist
        output_directory.mkdir(parents=True, exist_ok=True)
        
        # Get selected images from duplicate groups, or all processed images if no duplicates
        selected_images = self.db_manager.get_selected_images()
        
        # If no selected images (no duplicates found), consolidate all processed images
        if not selected_images:
            logger.info("No duplicates found, consolidating all processed images")
            selected_images = self.db_manager.get_all_processed_images()
        
        consolidation_results = {
            'total_files': len(selected_images),
            'successful_copies': 0,
            'failed_copies': 0,
            'skipped_files': 0,
            'errors': []
        }
        
        for image_record in selected_images:
            source_path = Path(image_record['file_path'])
            
            if not source_path.exists():
                logger.warning(f"Source file no longer exists: {source_path}")
                consolidation_results['skipped_files'] += 1
                continue
            
            try:
                # Determine target path
                if preserve_structure:
                    # Preserve subdirectory structure relative to source directories
                    relative_path = self._get_relative_path(source_path)
                    target_path = output_directory / relative_path
                else:
                    # Flatten structure - all files in output directory
                    target_path = output_directory / source_path.name
                
                # Create target directory if needed
                target_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Handle filename conflicts
                target_path = self._resolve_filename_conflict(target_path)
                
                # Copy or move the file
                if copy_mode:
                    import shutil
                    shutil.copy2(source_path, target_path)
                    logger.info(f"Copied: {source_path} -> {target_path}")
                else:
                    import shutil
                    shutil.move(str(source_path), str(target_path))
                    logger.info(f"Moved: {source_path} -> {target_path}")
                
                # Copy metadata if available
                self._copy_metadata(source_path, target_path)
                
                consolidation_results['successful_copies'] += 1
                
            except Exception as e:
                error_msg = f"Failed to process {source_path}: {e}"
                logger.error(error_msg)
                consolidation_results['errors'].append(error_msg)
                consolidation_results['failed_copies'] += 1
        
        logger.info(f"Consolidation completed: {consolidation_results['successful_copies']} "
                   f"successful, {consolidation_results['failed_copies']} failed")
        
        return consolidation_results
    
    def _get_relative_path(self, file_path: Path) -> Path:
        """Get relative path from source directories."""
        # Find which source directory this file belongs to
        for source_dir in self.source_directories:
            try:
                return file_path.relative_to(source_dir)
            except ValueError:
                continue
        
        # If not found in any source directory, use filename only
        return Path(file_path.name)
    
    def _resolve_filename_conflict(self, target_path: Path) -> Path:
        """Resolve filename conflicts by adding a suffix."""
        if not target_path.exists():
            return target_path
        
        # Add suffix to filename
        counter = 1
        while True:
            stem = target_path.stem
            suffix = target_path.suffix
            new_name = f"{stem}_{counter}{suffix}"
            new_path = target_path.parent / new_name
            
            if not new_path.exists():
                return new_path
            
            counter += 1
    
    def _copy_metadata(self, source_path: Path, target_path: Path) -> None:
        """Copy metadata from source to target file."""
        try:
            from utils.image_utils import copy_metadata
            copy_metadata(source_path, target_path)
        except Exception as e:
            logger.warning(f"Failed to copy metadata from {source_path} to {target_path}: {e}")
    
    def consolidate_all_files(self, output_directory: Path, preserve_structure: bool = True, 
                             copy_mode: bool = True, include_videos: bool = True,
                             include_corrupted: bool = False) -> Dict[str, Any]:
        """
        Consolidate all files (images, videos, corrupted) into a single output directory.
        
        Args:
            output_directory: Target directory for consolidated files
            preserve_structure: Whether to preserve subdirectory structure
            copy_mode: True to copy files, False to move files
            include_videos: Whether to include video files
            include_corrupted: Whether to include corrupted image files
            
        Returns:
            Dictionary with consolidation results
        """
        logger.info(f"Starting comprehensive file consolidation to {output_directory}")
        
        # Create output directory if it doesn't exist
        output_directory.mkdir(parents=True, exist_ok=True)
        
        consolidation_results = {
            'processed_images': {'total': 0, 'successful': 0, 'failed': 0},
            'video_files': {'total': 0, 'successful': 0, 'failed': 0},
            'corrupted_files': {'total': 0, 'successful': 0, 'failed': 0},
            'unprocessed_files': {'total': 0, 'successful': 0, 'failed': 0},
            'errors': []
        }
        
        # Get all files from source directories
        all_files = self._get_all_files_from_directories()
        
        # Process each file type
        for file_path, file_type in all_files:
            try:
                # Determine target path
                if preserve_structure:
                    relative_path = self._get_relative_path(file_path)
                    target_path = output_directory / relative_path
                else:
                    target_path = output_directory / file_path.name
                
                # Create target directory if needed
                target_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Handle filename conflicts
                target_path = self._resolve_filename_conflict(target_path)
                
                # Copy or move the file
                if copy_mode:
                    import shutil
                    shutil.copy2(file_path, target_path)
                    logger.info(f"Copied {file_type}: {file_path} -> {target_path}")
                else:
                    import shutil
                    shutil.move(str(file_path), str(target_path))
                    logger.info(f"Moved {file_type}: {file_path} -> {target_path}")
                
                consolidation_results[file_type]['successful'] += 1
                
            except Exception as e:
                error_msg = f"Failed to process {file_type} {file_path}: {e}"
                logger.error(error_msg)
                consolidation_results['errors'].append(error_msg)
                consolidation_results[file_type]['failed'] += 1
        
        # Log summary
        logger.info(f"Comprehensive consolidation completed:")
        for file_type, stats in consolidation_results.items():
            if file_type != 'errors':
                logger.info(f"  {file_type}: {stats['successful']}/{stats['total']} successful")
        
        return consolidation_results
    
    def _get_all_files_from_directories(self) -> List[Tuple[Path, str]]:
        """
        Get all files from source directories with their types.
        
        Returns:
            List of (file_path, file_type) tuples
        """
        all_files = []
        
        for source_dir in self.source_directories:
            if not source_dir.exists():
                continue
            
            # Get all files recursively
            for file_path in source_dir.rglob('*'):
                if file_path.is_file():
                    file_type = self._classify_file(file_path)
                    if file_type:
                        all_files.append((file_path, file_type))
        
        return all_files
    
    def _classify_file(self, file_path: Path) -> Optional[str]:
        """
        Classify a file based on its processing status and type.
        
        Args:
            file_path: Path to the file
            
        Returns:
            File type classification or None if should be skipped
        """
        # Check if it's a processed image
        if self.db_manager.get_image_by_path(str(file_path)):
            return 'processed_images'
        
        # Check if it's a video file
        if self._is_video_file(file_path):
            return 'video_files'
        
        # Check if it's a supported image format
        if self._is_supported_image_format(file_path):
            # If it's a supported format but not in database, it might be corrupted or unprocessed
            if self._is_valid_image_file(file_path):
                return 'unprocessed_files'  # Valid but not processed
            else:
                return 'corrupted_files'  # Corrupted image
        
        return None  # Skip unsupported files
    
    def _is_video_file(self, file_path: Path) -> bool:
        """Check if file is a video format."""
        from utils.image_utils import VIDEO_FORMATS
        return file_path.suffix.lower() in VIDEO_FORMATS
    
    def _is_supported_image_format(self, file_path: Path) -> bool:
        """Check if file is a supported image format."""
        from utils.image_utils import SUPPORTED_FORMATS
        return file_path.suffix.lower() in SUPPORTED_FORMATS
    
    def _is_valid_image_file(self, file_path: Path) -> bool:
        """Check if image file is valid."""
        from utils.image_utils import is_valid_image_file
        return is_valid_image_file(file_path)
    
    def get_file_summary(self) -> Dict[str, Any]:
        """
        Get a comprehensive summary of all files in source directories.
        
        Returns:
            Dictionary with file counts and details
        """
        all_files = self._get_all_files_from_directories()
        
        summary = {
            'processed_images': [],
            'video_files': [],
            'corrupted_files': [],
            'unprocessed_files': [],
            'total_files': len(all_files)
        }
        
        for file_path, file_type in all_files:
            summary[file_type].append(str(file_path))
        
        # Add counts
        for file_type in ['processed_images', 'video_files', 'corrupted_files', 'unprocessed_files']:
            summary[f'{file_type}_count'] = len(summary[file_type])
        
        return summary
    
    def test_similarity_detection(self, test_thresholds: List[float] = None) -> Dict[str, Any]:
        """
        Test similarity detection with different thresholds for debugging.
        
        Args:
            test_thresholds: List of thresholds to test (default: [0.5, 0.6, 0.7, 0.8, 0.9])
            
        Returns:
            Dictionary with results for each threshold
        """
        if test_thresholds is None:
            test_thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
        
        logger.info("Testing similarity detection with different thresholds")
        
        # Get all processed images
        all_images = self.db_manager.get_all_images()
        processed_images = [img for img in all_images if img['clip_embedding'] is not None]
        
        if len(processed_images) < 2:
            logger.warning("Not enough processed images to test similarity detection")
            return {}
        
        # Convert embeddings back to numpy arrays
        embeddings = []
        image_ids = []
        image_paths = []
        
        for img in processed_images:
            try:
                embedding = np.frombuffer(img['clip_embedding'], dtype=np.float32)
                embeddings.append(embedding)
                image_ids.append(img['id'])
                image_paths.append(img['file_path'])
            except Exception as e:
                logger.error(f"Failed to load embedding for image {img['id']}: {e}")
        
        results = {}
        
        for threshold in test_thresholds:
            logger.info(f"Testing threshold: {threshold}")
            
            duplicate_groups = []
            processed_indices = set()
            
            for i, embedding in enumerate(embeddings):
                if i in processed_indices:
                    continue
                
                # Find similar images
                similar_indices = self.clip_processor.find_similar_images(
                    embedding, embeddings, threshold
                )
                
                # Filter out already processed images
                similar_indices = [(idx, score) for idx, score in similar_indices 
                                 if idx not in processed_indices]
                
                if len(similar_indices) > 1:  # More than just the image itself
                    # Create duplicate group
                    group_images = []
                    for idx, score in similar_indices:
                        image_record = next(img for img in processed_images if img['id'] == image_ids[idx])
                        group_images.append({
                            'id': image_record['id'],
                            'file_path': image_record['file_path'],
                            'similarity_score': score
                        })
                        processed_indices.add(idx)
                    
                    duplicate_groups.append({
                        'images': group_images,
                        'threshold': threshold
                    })
            
            results[f'threshold_{threshold}'] = {
                'groups_found': len(duplicate_groups),
                'total_images_in_groups': sum(len(group['images']) for group in duplicate_groups),
                'groups': duplicate_groups
            }
            
            logger.info(f"  Threshold {threshold}: Found {len(duplicate_groups)} groups")
        
=======
"""
Main deduplication engine for Meme-Cleanup.

Orchestrates the entire deduplication process including scanning, embedding generation,
duplicate detection, and quality assessment.
"""

import logging
import time
import hashlib
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from joblib import Parallel, delayed

from db.database import DatabaseManager
from core.clip_processor import CLIPEmbeddingProcessor
from core.quality_metrics import QualityMetricsProcessor
from utils.image_utils import get_valid_image_files, extract_image_metadata, calculate_image_hash, load_image, is_valid_image_file, get_all_media_files, is_supported_image, is_video_file


logger = logging.getLogger(__name__)


@dataclass
class DeduplicationConfig:
    """Configuration for deduplication process."""
    similarity_threshold: float = 0.8
    batch_size: int = 8
    max_workers: int = 4
    use_gpu: bool = True
    quality_metric: str = "brisque"  # Default to BRISQUE for artifact detection
    save_progress: bool = True
    parallel_processing: bool = True
    n_jobs: int = -1  # Use all available cores


@dataclass
class DuplicateGroup:
    """Represents a group of duplicate images."""
    group_id: int
    images: List[Dict[str, Any]]
    similarity_threshold: float
    selected_image_id: Optional[int] = None


class Deduplicator:
    """Main deduplication engine."""
    
    def __init__(self, db_manager: DatabaseManager, config: DeduplicationConfig):
        """
        Initialize deduplicator.
        
        Args:
            db_manager: Database manager instance
            config: Deduplication configuration
        """
        self.db_manager = db_manager
        self.config = config
        
        # Initialize processors
        device = "cuda" if config.use_gpu else "cpu"
        self.clip_processor = CLIPEmbeddingProcessor(device=device)
        self.quality_processor = QualityMetricsProcessor()
        
        logger.info(f"Deduplicator initialized with config: {config}")
    
    def scan_directories(self, directories: List[Path]) -> List[Path]:
        """
        Scan directories for image files and store metadata using parallel processing.
        
        Args:
            directories: List of directories to scan
            
        Returns:
            List of discovered image file paths
        """
        logger.info(f"Scanning {len(directories)} directories for images")
        
        all_image_files = []
        processed_count = 0
        corrupted_files_count = 0
        skipped_files_count = 0
        start_time = time.time()
        
        for directory in directories:
            if not directory.exists():
                logger.warning(f"Directory does not exist: {directory}")
                continue
            
            logger.info(f"Scanning directory: {directory}")
            
            # Get all files first to see what we're working with
            all_files = list(directory.rglob('*'))
            image_files = get_valid_image_files(directory, recursive=True)
            
            logger.info(f"Directory {directory}:")
            logger.info(f"  Total files found: {len(all_files)}")
            logger.info(f"  Valid image files: {len(image_files)}")
            
            # Count corrupted files that were filtered out
            _, _, corrupted = get_all_media_files(directory, recursive=True)
            corrupted_files_count += len(corrupted)
            
            # Count skipped files (non-image, non-video)
            skipped_files = [f for f in all_files if f.is_file() and 
                           not is_supported_image(f) and 
                           not is_video_file(f)]
            skipped_files_count += len(skipped_files)
            
            logger.info(f"  Corrupted files: {len(corrupted)}")
            logger.info(f"  Skipped files: {len(skipped_files)}")
            
            if len(skipped_files) > 0:
                logger.info(f"  Sample skipped files:")
                for f in skipped_files[:10]:  # Show first 10
                    logger.info(f"    - {f.name}")
                if len(skipped_files) > 10:
                    logger.info(f"    ... and {len(skipped_files) - 10} more")
            
            # Process images in parallel if enabled
            if self.config.parallel_processing and len(image_files) > 10:
                logger.info(f"Using parallel processing for {len(image_files)} images")
                
                def process_single_image(image_path):
                    try:
                        # Double-check file validity before processing
                        if not is_valid_image_file(image_path):
                            logger.warning(f"Skipping corrupted file: {image_path}")
                            return None
                        
                        # Extract metadata
                        metadata = extract_image_metadata(image_path)
                        
                        # Calculate perceptual hash
                        image = load_image(image_path)
                        if image is not None:
                            metadata['perceptual_hash'] = calculate_image_hash(image)
                            metadata['processed_at'] = time.time()
                            return metadata
                    except Exception as e:
                        logger.error(f"Failed to process {image_path}: {e}")
                    return None
                
                # Process in parallel
                results = Parallel(n_jobs=self.config.n_jobs, verbose=1)(
                    delayed(process_single_image)(image_path) for image_path in image_files
                )
                
                # Store results in database
                successful_count = 0
                failed_count = 0
                for metadata in results:
                    if metadata:
                        try:
                            self.db_manager.insert_image(metadata)
                            successful_count += 1
                        except Exception as e:
                            logger.error(f"Failed to insert {metadata.get('file_path', 'unknown')}: {e}")
                            failed_count += 1
                    else:
                        failed_count += 1
                
                processed_count += successful_count
                logger.info(f"  Parallel processing results: {successful_count} successful, {failed_count} failed")
                
            else:
                # Sequential processing for small batches
                successful_count = 0
                failed_count = 0
                for image_path in image_files:
                    try:
                        # Extract metadata
                        metadata = extract_image_metadata(image_path)
                        
                        # Calculate perceptual hash
                        image = load_image(image_path)
                        if image is not None:
                            metadata['perceptual_hash'] = calculate_image_hash(image)
                            metadata['processed_at'] = time.time()
                            
                            # Store in database
                            self.db_manager.insert_image(metadata)
                            successful_count += 1
                        else:
                            failed_count += 1
                    
                    except Exception as e:
                        logger.error(f"Failed to process {image_path}: {e}")
                        failed_count += 1
                
                processed_count += successful_count
                logger.info(f"  Sequential processing results: {successful_count} successful, {failed_count} failed")
            
            all_image_files.extend(image_files)
        
        # Store scan results
        scan_results = {
            'directories_scanned': len(directories),
            'total_images_processed': processed_count,
            'total_images_found': len(all_image_files),
            'corrupted_files_skipped': corrupted_files_count,
            'skipped_files_count': skipped_files_count,
            'scan_duration': time.time() - start_time,
            'scan_completed_at': time.time()
        }
        
        self.db_manager.save_scan_results(scan_results)
        
        logger.info(f"Scan completed:")
        logger.info(f"  Total images found: {len(all_image_files)}")
        logger.info(f"  Successfully processed: {processed_count}")
        logger.info(f"  Corrupted files skipped: {corrupted_files_count}")
        logger.info(f"  Other files skipped: {skipped_files_count}")
        logger.info(f"  Scan duration: {scan_results['scan_duration']:.2f} seconds")
        
        return all_image_files
    
    def generate_embeddings(self, image_paths: Optional[List[Path]] = None) -> None:
        """
        Generate CLIP embeddings for images.
        
        Args:
            image_paths: List of image paths to process (None for all unprocessed)
        """
        if image_paths is None:
            # Get unprocessed images from database
            unprocessed_images = self.db_manager.get_unprocessed_images()
            image_paths = [Path(img['file_path']) for img in unprocessed_images]
        
        logger.info(f"Generating embeddings for {len(image_paths)} images")
        
        # Process in batches
        for i in range(0, len(image_paths), self.config.batch_size):
            batch_paths = image_paths[i:i + self.config.batch_size]
            
            try:
                # Generate embeddings
                embeddings = self.clip_processor.get_embeddings_batch(
                    batch_paths, self.config.batch_size
                )
                
                # Update database with embeddings and quality scores
                for j, (image_path, embedding) in enumerate(zip(batch_paths, embeddings)):
                    if embedding is not None:
                        # Get image record from database
                        image_record = self.db_manager.get_image_by_path(str(image_path))
                        if image_record:
                            # Calculate quality scores
                            quality_metrics = self.quality_processor.calculate_all_metrics(image_path)
                            
                            if quality_metrics:
                                brisque_score, niqe_score, combined_score = quality_metrics
                                
                                # Update database
                                self.db_manager.update_image_embeddings(
                                    image_record['id'],
                                    embedding.tobytes(),
                                    brisque_score,
                                    niqe_score
                                )
                
                logger.info(f"Processed batch {i//self.config.batch_size + 1}/{(len(image_paths) + self.config.batch_size - 1)//self.config.batch_size}")
                
            except Exception as e:
                logger.error(f"Failed to process batch {i//self.config.batch_size + 1}: {e}")
    
    def find_duplicates(self) -> List[DuplicateGroup]:
        """
        Find duplicate images using CLIP embeddings.
        
        Returns:
            List of duplicate groups
        """
        logger.info("Finding duplicate images")
        
        # Get all processed images
        all_images = self.db_manager.get_all_images()
        processed_images = [img for img in all_images if img['clip_embedding'] is not None]
        
        logger.info(f"Processing {len(processed_images)} images for duplicates")
        
        if len(processed_images) < 2:
            logger.warning("Not enough processed images to find duplicates (need at least 2)")
            return []
        
        # Convert embeddings back to numpy arrays
        embeddings = []
        image_ids = []
        image_paths = []
        
        for img in processed_images:
            try:
                embedding = np.frombuffer(img['clip_embedding'], dtype=np.float32)
                embeddings.append(embedding)
                image_ids.append(img['id'])
                image_paths.append(img['file_path'])
                logger.debug(f"Loaded embedding for: {img['file_path']}")
            except Exception as e:
                logger.error(f"Failed to load embedding for image {img['id']}: {e}")
        
        logger.info(f"Successfully loaded {len(embeddings)} embeddings")
        
        # Find duplicate groups
        duplicate_groups = []
        processed_indices = set()
        
        for i, embedding in enumerate(embeddings):
            if i in processed_indices:
                continue
            
            logger.debug(f"Checking for duplicates of: {image_paths[i]}")
            
            # Find similar images
            similar_indices = self.clip_processor.find_similar_images(
                embedding, embeddings, self.config.similarity_threshold
            )
            
            logger.debug(f"Found {len(similar_indices)} similar images for {image_paths[i]}")
            
            # Filter out already processed images
            similar_indices = [(idx, score) for idx, score in similar_indices 
                             if idx not in processed_indices]
            
            if len(similar_indices) > 1:  # More than just the image itself
                logger.info(f"Creating duplicate group with {len(similar_indices)} images")
                
                # Create duplicate group
                group_images = []
                for idx, score in similar_indices:
                    image_record = next(img for img in processed_images if img['id'] == image_ids[idx])
                    group_images.append({
                        'id': image_record['id'],
                        'file_path': image_record['file_path'],
                        'similarity_score': score,
                        'brisque_score': image_record['brisque_score'],
                        'niqe_score': image_record['niqe_score'],
                        'file_size': image_record['file_size'],
                        'width': image_record['width'],
                        'height': image_record['height']
                    })
                    processed_indices.add(idx)
                    logger.info(f"  Added to group: {image_record['file_path']} (score: {score:.3f})")
                
                # Sort by quality score (best first)
                group_images.sort(key=lambda x: x.get('brisque_score', float('inf')))
                
                # Create group in database
                group_hash = hashlib.md5(f"group_{len(duplicate_groups)}".encode()).hexdigest()
                group_id = self.db_manager.create_duplicate_group(group_hash, self.config.similarity_threshold)
                
                # Add images to group
                for img in group_images:
                    self.db_manager.add_image_to_duplicate_group(
                        group_id, img['id'], img['similarity_score']
                    )
                
                duplicate_groups.append(DuplicateGroup(
                    group_id=group_id,
                    images=group_images,
                    similarity_threshold=self.config.similarity_threshold
                ))
            else:
                logger.debug(f"No duplicates found for {image_paths[i]}")
        
        logger.info(f"Found {len(duplicate_groups)} duplicate groups")
        
        # Log some statistics about the embeddings
        if embeddings:
            # Calculate some basic statistics
            all_embeddings = np.array(embeddings)
            mean_similarity = np.mean(all_embeddings)
            std_similarity = np.std(all_embeddings)
            logger.info(f"Embedding statistics - Mean: {mean_similarity:.4f}, Std: {std_similarity:.4f}")
            logger.info(f"Similarity threshold: {self.config.similarity_threshold}")
        
        return duplicate_groups
    
    def select_best_images(self, duplicate_groups: List[DuplicateGroup]) -> None:
        """
        Automatically select the best quality image from each duplicate group.
        
        Args:
            duplicate_groups: List of duplicate groups
        """
        logger.info("Selecting best quality images from duplicate groups")
        
        for group in duplicate_groups:
            # Find image with best quality score
            best_image = None
            best_score = float('inf')
            
            for img in group.images:
                # Use combined quality score if available, otherwise BRISQUE
                quality_score = img.get('brisque_score', float('inf'))
                if img.get('niqe_score') is not None:
                    # Calculate combined score (simplified)
                    combined_score = (quality_score + img['niqe_score']) / 2
                    quality_score = combined_score
                
                if quality_score < best_score:
                    best_score = quality_score
                    best_image = img
            
            if best_image:
                # Mark as selected in database
                self.db_manager.mark_image_as_selected(
                    group.group_id, best_image['id'], True
                )
                group.selected_image_id = best_image['id']
                
                logger.info(f"Selected {best_image['file_path']} for group {group.group_id}")
    
    def run_full_deduplication(self, source_directories: List[Path]) -> List[DuplicateGroup]:
        """
        Run the complete deduplication process.
        
        Args:
            source_directories: List of source directories to process
            
        Returns:
            List of duplicate groups found
        """
        logger.info("Starting full deduplication process")
        
        try:
            # Step 1: Scan directories
            image_files = self.scan_directories(source_directories)
            
            # Step 2: Generate embeddings and quality scores
            self.generate_embeddings()
            
            # Step 3: Find duplicates
            duplicate_groups = self.find_duplicates()
            
            # Step 4: Select best images
            self.select_best_images(duplicate_groups)
            
            logger.info("Deduplication process completed successfully")
            return duplicate_groups
            
        except Exception as e:
            logger.error(f"Deduplication process failed: {e}")
            raise
    
    def get_duplicate_groups(self) -> List[DuplicateGroup]:
        """
        Get all duplicate groups from database.
        
        Returns:
            List of duplicate groups
        """
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
        duplicate_groups = []
        for group_data in groups_dict.values():
            duplicate_groups.append(DuplicateGroup(
                group_id=group_data['group_id'],
                images=group_data['images'],
                similarity_threshold=group_data['similarity_threshold'],
                selected_image_id=group_data['selected_image_id']
            ))
        
        return duplicate_groups
    
    def get_scan_results(self) -> Dict[str, Any]:
        """Get the latest scan results from the database."""
        try:
            cursor = self.db_manager._get_connection().cursor()
            cursor.execute("""
                SELECT * FROM scan_results 
                ORDER BY scan_completed_at DESC 
                LIMIT 1
            """)
            result = cursor.fetchone()
            
            if result:
                return {
                    'directories_scanned': result[1],
                    'total_images_processed': result[2],
                    'total_images_found': result[3],
                    'corrupted_files_skipped': result[4],
                    'skipped_files_count': result[5],
                    'scan_duration': result[6],
                    'scan_completed_at': result[7]
                }
            else:
                return {}
        except Exception as e:
            logger.error(f"Failed to get scan results: {e}")
            return {}
    
    def consolidate_files(self, output_directory: Path, preserve_structure: bool = True, 
                         copy_mode: bool = True) -> Dict[str, Any]:
        """
        Consolidate selected images into a single output directory.
        
        Args:
            output_directory: Target directory for consolidated files
            preserve_structure: Whether to preserve subdirectory structure
            copy_mode: True to copy files, False to move files
            
        Returns:
            Dictionary with consolidation results
        """
        logger.info(f"Starting file consolidation to {output_directory}")
        
        # Create output directory if it doesn't exist
        output_directory.mkdir(parents=True, exist_ok=True)
        
        # Get selected images from duplicate groups, or all processed images if no duplicates
        selected_images = self.db_manager.get_selected_images()
        
        # If no selected images (no duplicates found), consolidate all processed images
        if not selected_images:
            logger.info("No duplicates found, consolidating all processed images")
            selected_images = self.db_manager.get_all_processed_images()
        
        consolidation_results = {
            'total_files': len(selected_images),
            'successful_copies': 0,
            'failed_copies': 0,
            'skipped_files': 0,
            'errors': []
        }
        
        for image_record in selected_images:
            source_path = Path(image_record['file_path'])
            
            if not source_path.exists():
                logger.warning(f"Source file no longer exists: {source_path}")
                consolidation_results['skipped_files'] += 1
                continue
            
            try:
                # Determine target path
                if preserve_structure:
                    # Preserve subdirectory structure relative to source directories
                    relative_path = self._get_relative_path(source_path)
                    target_path = output_directory / relative_path
                else:
                    # Flatten structure - all files in output directory
                    target_path = output_directory / source_path.name
                
                # Create target directory if needed
                target_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Handle filename conflicts
                target_path = self._resolve_filename_conflict(target_path)
                
                # Copy or move the file
                if copy_mode:
                    import shutil
                    shutil.copy2(source_path, target_path)
                    logger.info(f"Copied: {source_path} -> {target_path}")
                else:
                    import shutil
                    shutil.move(str(source_path), str(target_path))
                    logger.info(f"Moved: {source_path} -> {target_path}")
                
                # Copy metadata if available
                self._copy_metadata(source_path, target_path)
                
                consolidation_results['successful_copies'] += 1
                
            except Exception as e:
                error_msg = f"Failed to process {source_path}: {e}"
                logger.error(error_msg)
                consolidation_results['errors'].append(error_msg)
                consolidation_results['failed_copies'] += 1
        
        logger.info(f"Consolidation completed: {consolidation_results['successful_copies']} "
                   f"successful, {consolidation_results['failed_copies']} failed")
        
        return consolidation_results
    
    def _get_relative_path(self, file_path: Path) -> Path:
        """Get relative path from source directories."""
        # Find which source directory this file belongs to
        for source_dir in self.source_directories:
            try:
                return file_path.relative_to(source_dir)
            except ValueError:
                continue
        
        # If not found in any source directory, use filename only
        return Path(file_path.name)
    
    def _resolve_filename_conflict(self, target_path: Path) -> Path:
        """Resolve filename conflicts by adding a suffix."""
        if not target_path.exists():
            return target_path
        
        # Add suffix to filename
        counter = 1
        while True:
            stem = target_path.stem
            suffix = target_path.suffix
            new_name = f"{stem}_{counter}{suffix}"
            new_path = target_path.parent / new_name
            
            if not new_path.exists():
                return new_path
            
            counter += 1
    
    def _copy_metadata(self, source_path: Path, target_path: Path) -> None:
        """Copy metadata from source to target file."""
        try:
            from utils.image_utils import copy_metadata
            copy_metadata(source_path, target_path)
        except Exception as e:
            logger.warning(f"Failed to copy metadata from {source_path} to {target_path}: {e}")
    
    def consolidate_all_files(self, output_directory: Path, preserve_structure: bool = True, 
                             copy_mode: bool = True, include_videos: bool = True,
                             include_corrupted: bool = False) -> Dict[str, Any]:
        """
        Consolidate all files (images, videos, corrupted) into a single output directory.
        
        Args:
            output_directory: Target directory for consolidated files
            preserve_structure: Whether to preserve subdirectory structure
            copy_mode: True to copy files, False to move files
            include_videos: Whether to include video files
            include_corrupted: Whether to include corrupted image files
            
        Returns:
            Dictionary with consolidation results
        """
        logger.info(f"Starting comprehensive file consolidation to {output_directory}")
        
        # Create output directory if it doesn't exist
        output_directory.mkdir(parents=True, exist_ok=True)
        
        consolidation_results = {
            'processed_images': {'total': 0, 'successful': 0, 'failed': 0},
            'video_files': {'total': 0, 'successful': 0, 'failed': 0},
            'corrupted_files': {'total': 0, 'successful': 0, 'failed': 0},
            'unprocessed_files': {'total': 0, 'successful': 0, 'failed': 0},
            'errors': []
        }
        
        # Get all files from source directories
        all_files = self._get_all_files_from_directories()
        
        # Process each file type
        for file_path, file_type in all_files:
            try:
                # Determine target path
                if preserve_structure:
                    relative_path = self._get_relative_path(file_path)
                    target_path = output_directory / relative_path
                else:
                    target_path = output_directory / file_path.name
                
                # Create target directory if needed
                target_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Handle filename conflicts
                target_path = self._resolve_filename_conflict(target_path)
                
                # Copy or move the file
                if copy_mode:
                    import shutil
                    shutil.copy2(file_path, target_path)
                    logger.info(f"Copied {file_type}: {file_path} -> {target_path}")
                else:
                    import shutil
                    shutil.move(str(file_path), str(target_path))
                    logger.info(f"Moved {file_type}: {file_path} -> {target_path}")
                
                consolidation_results[file_type]['successful'] += 1
                
            except Exception as e:
                error_msg = f"Failed to process {file_type} {file_path}: {e}"
                logger.error(error_msg)
                consolidation_results['errors'].append(error_msg)
                consolidation_results[file_type]['failed'] += 1
        
        # Log summary
        logger.info(f"Comprehensive consolidation completed:")
        for file_type, stats in consolidation_results.items():
            if file_type != 'errors':
                logger.info(f"  {file_type}: {stats['successful']}/{stats['total']} successful")
        
        return consolidation_results
    
    def _get_all_files_from_directories(self) -> List[Tuple[Path, str]]:
        """
        Get all files from source directories with their types.
        
        Returns:
            List of (file_path, file_type) tuples
        """
        all_files = []
        
        for source_dir in self.source_directories:
            if not source_dir.exists():
                continue
            
            # Get all files recursively
            for file_path in source_dir.rglob('*'):
                if file_path.is_file():
                    file_type = self._classify_file(file_path)
                    if file_type:
                        all_files.append((file_path, file_type))
        
        return all_files
    
    def _classify_file(self, file_path: Path) -> Optional[str]:
        """
        Classify a file based on its processing status and type.
        
        Args:
            file_path: Path to the file
            
        Returns:
            File type classification or None if should be skipped
        """
        # Check if it's a processed image
        if self.db_manager.get_image_by_path(str(file_path)):
            return 'processed_images'
        
        # Check if it's a video file
        if self._is_video_file(file_path):
            return 'video_files'
        
        # Check if it's a supported image format
        if self._is_supported_image_format(file_path):
            # If it's a supported format but not in database, it might be corrupted or unprocessed
            if self._is_valid_image_file(file_path):
                return 'unprocessed_files'  # Valid but not processed
            else:
                return 'corrupted_files'  # Corrupted image
        
        return None  # Skip unsupported files
    
    def _is_video_file(self, file_path: Path) -> bool:
        """Check if file is a video format."""
        from utils.image_utils import VIDEO_FORMATS
        return file_path.suffix.lower() in VIDEO_FORMATS
    
    def _is_supported_image_format(self, file_path: Path) -> bool:
        """Check if file is a supported image format."""
        from utils.image_utils import SUPPORTED_FORMATS
        return file_path.suffix.lower() in SUPPORTED_FORMATS
    
    def _is_valid_image_file(self, file_path: Path) -> bool:
        """Check if image file is valid."""
        from utils.image_utils import is_valid_image_file
        return is_valid_image_file(file_path)
    
    def get_file_summary(self) -> Dict[str, Any]:
        """
        Get a comprehensive summary of all files in source directories.
        
        Returns:
            Dictionary with file counts and details
        """
        all_files = self._get_all_files_from_directories()
        
        summary = {
            'processed_images': [],
            'video_files': [],
            'corrupted_files': [],
            'unprocessed_files': [],
            'total_files': len(all_files)
        }
        
        for file_path, file_type in all_files:
            summary[file_type].append(str(file_path))
        
        # Add counts
        for file_type in ['processed_images', 'video_files', 'corrupted_files', 'unprocessed_files']:
            summary[f'{file_type}_count'] = len(summary[file_type])
        
        return summary
    
    def test_similarity_detection(self, test_thresholds: List[float] = None) -> Dict[str, Any]:
        """
        Test similarity detection with different thresholds for debugging.
        
        Args:
            test_thresholds: List of thresholds to test (default: [0.5, 0.6, 0.7, 0.8, 0.9])
            
        Returns:
            Dictionary with results for each threshold
        """
        if test_thresholds is None:
            test_thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
        
        logger.info("Testing similarity detection with different thresholds")
        
        # Get all processed images
        all_images = self.db_manager.get_all_images()
        processed_images = [img for img in all_images if img['clip_embedding'] is not None]
        
        if len(processed_images) < 2:
            logger.warning("Not enough processed images to test similarity detection")
            return {}
        
        # Convert embeddings back to numpy arrays
        embeddings = []
        image_ids = []
        image_paths = []
        
        for img in processed_images:
            try:
                embedding = np.frombuffer(img['clip_embedding'], dtype=np.float32)
                embeddings.append(embedding)
                image_ids.append(img['id'])
                image_paths.append(img['file_path'])
            except Exception as e:
                logger.error(f"Failed to load embedding for image {img['id']}: {e}")
        
        results = {}
        
        for threshold in test_thresholds:
            logger.info(f"Testing threshold: {threshold}")
            
            duplicate_groups = []
            processed_indices = set()
            
            for i, embedding in enumerate(embeddings):
                if i in processed_indices:
                    continue
                
                # Find similar images
                similar_indices = self.clip_processor.find_similar_images(
                    embedding, embeddings, threshold
                )
                
                # Filter out already processed images
                similar_indices = [(idx, score) for idx, score in similar_indices 
                                 if idx not in processed_indices]
                
                if len(similar_indices) > 1:  # More than just the image itself
                    # Create duplicate group
                    group_images = []
                    for idx, score in similar_indices:
                        image_record = next(img for img in processed_images if img['id'] == image_ids[idx])
                        group_images.append({
                            'id': image_record['id'],
                            'file_path': image_record['file_path'],
                            'similarity_score': score
                        })
                        processed_indices.add(idx)
                    
                    duplicate_groups.append({
                        'images': group_images,
                        'threshold': threshold
                    })
            
            results[f'threshold_{threshold}'] = {
                'groups_found': len(duplicate_groups),
                'total_images_in_groups': sum(len(group['images']) for group in duplicate_groups),
                'groups': duplicate_groups
            }
            
            logger.info(f"  Threshold {threshold}: Found {len(duplicate_groups)} groups")
        
>>>>>>> 88a757b8ecc4ae4d819ae02a34c56b2a1ebc3714
        return results 