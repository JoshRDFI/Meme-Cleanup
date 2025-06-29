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
    quality_metric: str = "BRISQUE"  # Default to BRISQUE for artifact detection
    save_progress: bool = True
    parallel_processing: bool = True
    n_jobs: int = -1  # Use all available cores
    skip_corrupted: bool = True
    skip_animated: bool = False
    min_file_size: int = 1000


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
        self.scan_results = {}
        
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
        scan_duration = time.time() - start_time
        self.scan_results = {
            'directories_scanned': len(directories),
            'total_images_found': len(all_image_files),
            'total_images_processed': processed_count,
            'corrupted_files_skipped': corrupted_files_count,
            'skipped_files_count': skipped_files_count,
            'scan_duration': scan_duration,
            'scan_completed_at': time.time()
        }
        
        # Save scan results to database
        self.db_manager.save_scan_results(self.scan_results)
        
        logger.info(f"Scan completed: {processed_count} images processed in {scan_duration:.2f} seconds")
        return all_image_files
    
    def generate_embeddings(self, image_paths: Optional[List[Path]] = None) -> None:
        """
        Generate CLIP embeddings for images.
        
        Args:
            image_paths: Optional list of specific image paths to process
        """
        if image_paths is None:
            # Get unprocessed images from database
            unprocessed_images = self.db_manager.get_unprocessed_images()
            image_paths = [Path(img['file_path']) for img in unprocessed_images]
        
        if not image_paths:
            logger.info("No images to process for embeddings")
            return
        
        logger.info(f"Generating embeddings for {len(image_paths)} images")
        
        # Process in batches
        for i in range(0, len(image_paths), self.config.batch_size):
            batch_paths = image_paths[i:i + self.config.batch_size]
            logger.info(f"Processing batch {i//self.config.batch_size + 1}/{(len(image_paths) + self.config.batch_size - 1)//self.config.batch_size}")
            
            # Generate embeddings
            embeddings = self.clip_processor.get_embeddings_batch(batch_paths, self.config.batch_size)
            
            # Process each image in batch
            for j, (image_path, embedding) in enumerate(zip(batch_paths, embeddings)):
                if embedding is None:
                    logger.warning(f"Failed to generate embedding for {image_path}")
                    continue
                
                try:
                    # Get image record from database
                    image_record = self.db_manager.get_image_by_path(str(image_path))
                    if not image_record:
                        logger.warning(f"Image not found in database: {image_path}")
                        continue
                    
                    # Generate quality metrics
                    image = load_image(image_path)
                    if image is not None:
                        brisque_score = self.quality_processor.calculate_brisque(image)
                        niqe_score = self.quality_processor.calculate_niqe(image)
                        
                        # Update database with embeddings and quality scores
                        self.db_manager.update_image_embeddings(
                            image_record['id'],
                            embedding.tobytes(),
                            brisque_score,
                            niqe_score
                        )
                        
                        logger.debug(f"Updated embeddings for {image_path}")
                    
                except Exception as e:
                    logger.error(f"Failed to update embeddings for {image_path}: {e}")
        
        logger.info("Embedding generation completed")
    
    def find_duplicates(self) -> List[DuplicateGroup]:
        """
        Find duplicate images using CLIP embeddings.
        
        Returns:
            List of duplicate groups
        """
        logger.info("Finding duplicate images")
        
        # Get all processed images
        processed_images = self.db_manager.get_all_processed_images()
        if len(processed_images) < 2:
            logger.info("Not enough processed images to find duplicates")
            return []
        
        # Group images by perceptual hash first (exact duplicates)
        hash_groups = {}
        for image in processed_images:
            if image.get('perceptual_hash'):
                hash_groups.setdefault(image['perceptual_hash'], []).append(image)
        
        # Find exact duplicates
        exact_duplicates = [images for images in hash_groups.values() if len(images) > 1]
        logger.info(f"Found {len(exact_duplicates)} groups of exact duplicates")
        
        # Find similar images using CLIP embeddings
        similar_groups = self._find_similar_images(processed_images)
        logger.info(f"Found {len(similar_groups)} groups of similar images")
        
        # Combine and create duplicate groups
        all_groups = []
        group_id = 1
        
        # Add exact duplicates
        for images in exact_duplicates:
            group = DuplicateGroup(
                group_id=group_id,
                images=images,
                similarity_threshold=1.0
            )
            all_groups.append(group)
            group_id += 1
        
        # Add similar images
        for images in similar_groups:
            group = DuplicateGroup(
                group_id=group_id,
                images=images,
                similarity_threshold=self.config.similarity_threshold
            )
            all_groups.append(group)
            group_id += 1
        
        # Store groups in database
        for group in all_groups:
            group_hash = hashlib.md5(f"group_{group.group_id}".encode()).hexdigest()
            db_group_id = self.db_manager.create_duplicate_group(group_hash, group.similarity_threshold)
            
            for image in group.images:
                self.db_manager.add_image_to_duplicate_group(db_group_id, image['id'], 1.0)
        
        logger.info(f"Total duplicate groups found: {len(all_groups)}")
        return all_groups
    
    def _find_similar_images(self, images: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """
        Find similar images using CLIP embeddings.
        
        Args:
            images: List of image records
            
        Returns:
            List of image groups
        """
        similar_groups = []
        processed = set()
        
        for i, image1 in enumerate(images):
            if image1['id'] in processed:
                continue
            
            # Get embedding for image1
            if not image1.get('clip_embedding'):
                continue
            
            embedding1 = np.frombuffer(image1['clip_embedding'], dtype=np.float32)
            
            # Find similar images
            similar_images = [image1]
            processed.add(image1['id'])
            
            for j, image2 in enumerate(images[i+1:], i+1):
                if image2['id'] in processed:
                    continue
                
                if not image2.get('clip_embedding'):
                    continue
                
                embedding2 = np.frombuffer(image2['clip_embedding'], dtype=np.float32)
                similarity = self.clip_processor.calculate_similarity(embedding1, embedding2)
                
                if similarity >= self.config.similarity_threshold:
                    similar_images.append(image2)
                    processed.add(image2['id'])
            
            # Only create group if we have multiple similar images
            if len(similar_images) > 1:
                similar_groups.append(similar_images)
        
        return similar_groups
    
    def select_best_images(self, duplicate_groups: List[DuplicateGroup]) -> None:
        """
        Select the best image from each duplicate group based on quality metrics.
        
        Args:
            duplicate_groups: List of duplicate groups
        """
        logger.info(f"Selecting best images from {len(duplicate_groups)} groups")
        
        for group in duplicate_groups:
            if not group.images:
                continue
            
            # Sort images by quality score
            scored_images = []
            for image in group.images:
                if self.config.quality_metric == "BRISQUE":
                    score = image.get('brisque_score', float('inf'))
                elif self.config.quality_metric == "NIQE":
                    score = image.get('niqe_score', float('inf'))
                else:  # Combined
                    brisque = image.get('brisque_score', float('inf'))
                    niqe = image.get('niqe_score', float('inf'))
                    score = (brisque + niqe) / 2
                
                scored_images.append((image, score))
            
            # Sort by score (lower is better for BRISQUE/NIQE)
            scored_images.sort(key=lambda x: x[1])
            
            # Select the best image
            best_image = scored_images[0][0]
            group.selected_image_id = best_image['id']
            
            # Mark as selected in database
            db_groups = self.db_manager.get_duplicate_groups()
            for db_group in db_groups:
                if any(img['id'] == best_image['id'] for img in db_group['images']):
                    self.db_manager.mark_image_as_selected(db_group['id'], best_image['id'], True)
                    break
        
        logger.info("Best image selection completed")
    
    def run_full_deduplication(self, source_directories: List[Path]) -> List[DuplicateGroup]:
        """
        Run the complete deduplication process.
        
        Args:
            source_directories: List of source directories to scan
            
        Returns:
            List of duplicate groups
        """
        logger.info("Starting full deduplication process")
        
        # Step 1: Scan directories
        image_files = self.scan_directories(source_directories)
        
        # Step 2: Generate embeddings
        self.generate_embeddings()
        
        # Step 3: Find duplicates
        duplicate_groups = self.find_duplicates()
        
        # Step 4: Select best images
        self.select_best_images(duplicate_groups)
        
        logger.info("Full deduplication process completed")
        return duplicate_groups
    
    def get_duplicate_groups(self) -> List[DuplicateGroup]:
        """
        Get duplicate groups from database.
        
        Returns:
            List of duplicate groups
        """
        db_groups = self.db_manager.get_duplicate_groups()
        
        groups = []
        for db_group in db_groups:
            group = DuplicateGroup(
                group_id=db_group['id'],
                images=db_group['images'],
                similarity_threshold=db_group['similarity_threshold']
            )
            
            # Find selected image
            for image in group.images:
                if image.get('is_selected'):
                    group.selected_image_id = image['id']
                    break
            
            groups.append(group)
        
        return groups
    
    def get_scan_results(self) -> Dict[str, Any]:
        """
        Get scan results.
        
        Returns:
            Dictionary with scan statistics
        """
        return self.scan_results.copy()
    
    def consolidate_files(self, output_directory: Path, preserve_structure: bool = True, 
                         copy_mode: bool = True) -> Dict[str, Any]:
        """
        Consolidate selected files to output directory.
        
        Args:
            output_directory: Output directory for consolidated files
            preserve_structure: Whether to preserve directory structure
            copy_mode: Whether to copy (True) or move (False) files
            
        Returns:
            Dictionary with consolidation results
        """
        logger.info(f"Consolidating files to {output_directory}")
        
        # Create output directory
        output_directory.mkdir(parents=True, exist_ok=True)
        
        # Get selected images
        selected_images = self.db_manager.get_selected_images()
        
        if not selected_images:
            logger.warning("No images selected for consolidation")
            return {'consolidated_count': 0, 'errors': []}
        
        consolidated_count = 0
        errors = []
        
        for image in selected_images:
            try:
                source_path = Path(image['file_path'])
                if not source_path.exists():
                    logger.warning(f"Source file does not exist: {source_path}")
                    continue
                
                # Determine target path
                if preserve_structure:
                    # Preserve relative path structure
                    relative_path = self._get_relative_path(source_path)
                    target_path = output_directory / relative_path
                else:
                    # Flat structure
                    target_path = output_directory / source_path.name
                
                # Resolve filename conflicts
                target_path = self._resolve_filename_conflict(target_path)
                
                # Create target directory
                target_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Copy or move file
                if copy_mode:
                    import shutil
                    shutil.copy2(source_path, target_path)
                else:
                    import shutil
                    shutil.move(str(source_path), str(target_path))
                
                # Copy metadata
                self._copy_metadata(source_path, target_path)
                
                consolidated_count += 1
                logger.debug(f"Consolidated: {source_path} -> {target_path}")
                
            except Exception as e:
                error_msg = f"Failed to consolidate {image.get('file_path', 'unknown')}: {e}"
                logger.error(error_msg)
                errors.append(error_msg)
        
        logger.info(f"Consolidation completed: {consolidated_count} files processed")
        return {
            'consolidated_count': consolidated_count,
            'errors': errors
        }
    
    def _get_relative_path(self, file_path: Path) -> Path:
        """
        Get relative path from source directories.
        
        Args:
            file_path: File path
            
        Returns:
            Relative path
        """
        # This is a simplified version - in practice you'd need to track source directories
        return file_path.name
    
    def _resolve_filename_conflict(self, target_path: Path) -> Path:
        """
        Resolve filename conflicts by adding suffix.
        
        Args:
            target_path: Target file path
            
        Returns:
            Resolved file path
        """
        if not target_path.exists():
            return target_path
        
        counter = 1
        while True:
            stem = target_path.stem
            suffix = target_path.suffix
            new_path = target_path.parent / f"{stem}_{counter}{suffix}"
            
            if not new_path.exists():
                return new_path
            
            counter += 1
    
    def _copy_metadata(self, source_path: Path, target_path: Path) -> None:
        """
        Copy metadata from source to target file.
        
        Args:
            source_path: Source file path
            target_path: Target file path
        """
        try:
            from utils.image_utils import copy_metadata
            copy_metadata(source_path, target_path)
        except Exception as e:
            logger.warning(f"Failed to copy metadata: {e}")
    
    def get_file_summary(self) -> Dict[str, Any]:
        """
        Get summary of all files in source directories.
        
        Returns:
            Dictionary with file summary
        """
        # This would need to be implemented based on your source directories
        return {
            'total_files': 0,
            'image_files': 0,
            'video_files': 0,
            'other_files': 0
        }
    
    def test_similarity_detection(self, test_thresholds: List[float] = None) -> Dict[str, Any]:
        """
        Test similarity detection with different thresholds.
        
        Args:
            test_thresholds: List of thresholds to test
            
        Returns:
            Dictionary with test results
        """
        if test_thresholds is None:
            test_thresholds = [0.7, 0.8, 0.9, 0.95]
        
        logger.info(f"Testing similarity detection with thresholds: {test_thresholds}")
        
        results = {}
        processed_images = self.db_manager.get_all_processed_images()
        
        for threshold in test_thresholds:
            # Temporarily set threshold
            original_threshold = self.config.similarity_threshold
            self.config.similarity_threshold = threshold
            
            # Find duplicates
            duplicate_groups = self.find_duplicates()
            
            results[threshold] = {
                'groups_found': len(duplicate_groups),
                'total_duplicates': sum(len(group.images) for group in duplicate_groups)
            }
            
            # Restore original threshold
            self.config.similarity_threshold = original_threshold
        
        return results 