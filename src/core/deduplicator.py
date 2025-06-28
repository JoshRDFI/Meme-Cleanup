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
from core.clip_processor import CLIPProcessor
from core.quality_metrics import QualityMetricsProcessor
from utils.image_utils import get_image_files, extract_image_metadata, calculate_image_hash, load_image


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
        self.clip_processor = CLIPProcessor(device=device)
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
        
        for directory in directories:
            if not directory.exists():
                logger.warning(f"Directory does not exist: {directory}")
                continue
            
            image_files = get_image_files(directory, recursive=True)
            logger.info(f"Found {len(image_files)} images in {directory}")
            
            # Process images in parallel if enabled
            if self.config.parallel_processing and len(image_files) > 10:
                logger.info(f"Using parallel processing for {len(image_files)} images")
                
                def process_single_image(image_path):
                    try:
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
                for metadata in results:
                    if metadata:
                        self.db_manager.insert_image(metadata)
                        processed_count += 1
                        
                        if processed_count % 100 == 0:
                            logger.info(f"Processed {processed_count} images...")
            else:
                # Sequential processing for small batches
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
                            processed_count += 1
                            
                            if processed_count % 100 == 0:
                                logger.info(f"Processed {processed_count} images...")
                    
                    except Exception as e:
                        logger.error(f"Failed to process {image_path}: {e}")
            
            all_image_files.extend(image_files)
        
        logger.info(f"Scan complete. Total images found: {len(all_image_files)}")
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
        
        # Convert embeddings back to numpy arrays
        embeddings = []
        image_ids = []
        
        for img in processed_images:
            try:
                embedding = np.frombuffer(img['clip_embedding'], dtype=np.float32)
                embeddings.append(embedding)
                image_ids.append(img['id'])
            except Exception as e:
                logger.error(f"Failed to load embedding for image {img['id']}: {e}")
        
        # Find duplicate groups
        duplicate_groups = []
        processed_indices = set()
        
        for i, embedding in enumerate(embeddings):
            if i in processed_indices:
                continue
            
            # Find similar images
            similar_indices = self.clip_processor.find_similar_images(
                embedding, embeddings, self.config.similarity_threshold
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
                        'similarity_score': score,
                        'brisque_score': image_record['brisque_score'],
                        'niqe_score': image_record['niqe_score'],
                        'file_size': image_record['file_size'],
                        'width': image_record['width'],
                        'height': image_record['height']
                    })
                    processed_indices.add(idx)
                
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
        
        logger.info(f"Found {len(duplicate_groups)} duplicate groups")
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