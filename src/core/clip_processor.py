"""
CLIP processor for Meme-Cleanup.

Handles loading and running the CLIP model to generate image embeddings for similarity comparison.
"""

import logging
import torch
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

from utils.image_utils import load_image, preprocess_image_for_clip


logger = logging.getLogger(__name__)


class CLIPProcessor:
    """Handles CLIP model operations for image embedding generation."""
    
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device: Optional[str] = None):
        """
        Initialize CLIP processor.
        
        Args:
            model_name: CLIP model name to use
            device: Device to run model on ('cuda', 'cpu', or None for auto-detection)
        """
        self.model_name = model_name
        self.device = self._get_device(device)
        
        logger.info(f"Initializing CLIP model '{model_name}' on device '{self.device}'")
        
        try:
            # Load CLIP model and processor
            self.model = CLIPModel.from_pretrained(model_name)
            self.processor = CLIPProcessor.from_pretrained(model_name)
            
            # Move model to device
            self.model.to(self.device)
            self.model.eval()
            
            logger.info("CLIP model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load CLIP model: {e}")
            raise
    
    def _get_device(self, device: Optional[str]) -> str:
        """
        Determine the best device to use for model inference.
        
        Args:
            device: User-specified device or None for auto-detection
            
        Returns:
            Device string ('cuda' or 'cpu')
        """
        if device:
            return device
        
        # Auto-detect CUDA availability
        if torch.cuda.is_available():
            logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
            return 'cuda'
        else:
            logger.info("CUDA not available, using CPU")
            return 'cpu'
    
    def get_embedding(self, image_path: Path) -> Optional[np.ndarray]:
        """
        Generate CLIP embedding for an image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            CLIP embedding as numpy array or None if failed
        """
        try:
            # Load and preprocess image
            image = load_image(image_path)
            if image is None:
                return None
            
            # Convert to PIL Image for CLIP processor
            pil_image = Image.fromarray(image)
            
            # Process image with CLIP
            with torch.no_grad():
                inputs = self.processor(images=pil_image, return_tensors="pt", padding=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Get image features
                image_features = self.model.get_image_features(**inputs)
                
                # Normalize features
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                
                # Convert to numpy array
                embedding = image_features.cpu().numpy().flatten()
                
                return embedding
                
        except Exception as e:
            logger.error(f"Failed to generate CLIP embedding for {image_path}: {e}")
            return None
    
    def get_embeddings_batch(self, image_paths: List[Path], batch_size: int = 8) -> List[Optional[np.ndarray]]:
        """
        Generate CLIP embeddings for multiple images in batches.
        
        Args:
            image_paths: List of image file paths
            batch_size: Number of images to process in each batch
            
        Returns:
            List of CLIP embeddings (None for failed images)
        """
        embeddings = []
        
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            batch_images = []
            valid_indices = []
            
            # Load batch images
            for j, path in enumerate(batch_paths):
                image = load_image(path)
                if image is not None:
                    batch_images.append(Image.fromarray(image))
                    valid_indices.append(j)
            
            if not batch_images:
                # All images in batch failed
                embeddings.extend([None] * len(batch_paths))
                continue
            
            try:
                # Process batch with CLIP
                with torch.no_grad():
                    inputs = self.processor(images=batch_images, return_tensors="pt", padding=True)
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    
                    # Get image features
                    image_features = self.model.get_image_features(**inputs)
                    
                    # Normalize features
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                    
                    # Convert to numpy arrays
                    batch_embeddings = image_features.cpu().numpy()
                    
                    # Create result list with None for failed images
                    batch_results = [None] * len(batch_paths)
                    for idx, embedding in zip(valid_indices, batch_embeddings):
                        batch_results[idx] = embedding.flatten()
                    
                    embeddings.extend(batch_results)
                    
            except Exception as e:
                logger.error(f"Failed to process batch {i//batch_size + 1}: {e}")
                embeddings.extend([None] * len(batch_paths))
        
        return embeddings
    
    def calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two CLIP embeddings.
        
        Args:
            embedding1: First CLIP embedding
            embedding2: Second CLIP embedding
            
        Returns:
            Similarity score between 0 and 1
        """
        try:
            # Calculate cosine similarity
            similarity = np.dot(embedding1, embedding2) / (
                np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
            )
            return float(similarity)
        except Exception as e:
            logger.error(f"Failed to calculate similarity: {e}")
            return 0.0
    
    def find_similar_images(self, target_embedding: np.ndarray, 
                          candidate_embeddings: List[np.ndarray],
                          threshold: float = 0.8) -> List[Tuple[int, float]]:
        """
        Find images similar to a target image.
        
        Args:
            target_embedding: CLIP embedding of the target image
            candidate_embeddings: List of candidate embeddings
            threshold: Similarity threshold (0-1)
            
        Returns:
            List of (index, similarity_score) tuples for similar images
        """
        similar_images = []
        
        for i, candidate_embedding in enumerate(candidate_embeddings):
            if candidate_embedding is not None:
                similarity = self.calculate_similarity(target_embedding, candidate_embedding)
                if similarity >= threshold:
                    similar_images.append((i, similarity))
        
        # Sort by similarity score (descending)
        similar_images.sort(key=lambda x: x[1], reverse=True)
        return similar_images 