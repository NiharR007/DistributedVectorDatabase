import faiss
import numpy as np
import yaml
import logging
from typing import List, Tuple, Optional, Union
import os
import time

class ShardNode:
    def __init__(self, config: Union[str, dict]):
        """Initialize a shard node with FAISS index.
        
        Args:
            config: Either a path to shard node configuration file or a config dictionary
        """
        self.load_config(config)
        self.setup_logging()
        self.dimension = 512  # Vector dimension
        self.index = None
        self.initialize_index()
        
    def load_config(self, config: Union[str, dict]):
        """Load configuration from yaml file or dict."""
        if isinstance(config, str):
            with open(config, 'r') as f:
                config = yaml.safe_load(f)
            
        self.storage_path = config['storage_path']
        self.monitoring = config['monitoring']
        
        os.makedirs(self.storage_path, exist_ok=True)
        
    def setup_logging(self):
        """Setup logging configuration."""
        log_level = getattr(logging, self.monitoring['log_level'])
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def initialize_index(self):
        """Initialize FAISS index for vector storage and search."""
        logging.info("Initializing FlatL2 index with IDMap")
        # Create base index
        base_index = faiss.IndexFlatL2(self.dimension)
        # Wrap with IDMap to support custom IDs
        self.index = faiss.IndexIDMap(base_index)
        self.next_id = 0  # Track next available ID if none provided
        
    def add_vectors(self, vectors: np.ndarray, ids: Optional[np.ndarray] = None):
        """Add vectors to the shard.
        
        Args:
            vectors: Vectors to add (n_vectors, dimension)
            ids: Optional vector IDs
        """
        try:
            self.logger.info(f"Adding vectors: shape={vectors.shape}, dtype={vectors.dtype}")
            
            if vectors.ndim == 1:
                vectors = vectors.reshape(1, -1)
                self.logger.debug("Reshaped single vector to 2D array")
                
            if vectors.dtype != np.float32:
                self.logger.debug(f"Converting vectors from {vectors.dtype} to float32")
                vectors = vectors.astype(np.float32)
                
            n_vectors = vectors.shape[0]
            
            # Check if dimensions match
            if vectors.shape[1] != self.dimension:
                self.logger.error(f"Vector dimension mismatch: expected {self.dimension}, got {vectors.shape[1]}")
                raise ValueError(f"Vector dimension mismatch: expected {self.dimension}, got {vectors.shape[1]}")
            
            # Handle IDs
            if ids is None:
                self.logger.debug(f"No IDs provided, generating sequential IDs starting from {self.next_id}")
                ids = np.arange(self.next_id, self.next_id + n_vectors, dtype=np.int64)
                self.next_id += n_vectors
            elif ids.dtype != np.int64:
                self.logger.debug(f"Converting IDs from {ids.dtype} to int64")
                ids = ids.astype(np.int64)
                
            self.logger.info(f"Adding {n_vectors} vectors with IDs range: {ids[0]} to {ids[-1]}")
            
            # Add vectors to the index
            self.index.add_with_ids(vectors, ids)
            
            # Verify vectors were added
            ntotal_after = self.index.ntotal
            self.logger.info(f"Index now contains {ntotal_after} total vectors")
            
            return ntotal_after
        except Exception as e:
            self.logger.error(f"Error adding vectors: {str(e)}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            raise
            
    def search(self, query_vectors: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Search for nearest neighbors.
        
        Args:
            query_vectors: Query vectors (n_queries, dimension)
            k: Number of nearest neighbors to return
            
        Returns:
            Tuple of (distances, indices)
        """
        self.logger.debug(f"Starting search with query_vectors shape={query_vectors.shape}, k={k}")
        self.logger.debug(f"Index has {self.index.ntotal} total vectors")
        
        if query_vectors.dtype != np.float32:
            query_vectors = query_vectors.astype(np.float32)
            
        # Check if we have enough vectors in the index
        if self.index.ntotal < k:
            self.logger.warning(f"Index only has {self.index.ntotal} vectors, but k={k}")
            k = max(1, self.index.ntotal)  # Ensure k is at least 1 if we have any vectors
            
        if self.index.ntotal == 0:
            self.logger.warning("Index is empty, returning empty results")
            return np.empty((query_vectors.shape[0], 0), dtype=np.float32), np.empty((query_vectors.shape[0], 0), dtype=np.int64)
            
        self.logger.debug(f"Searching with adjusted k={k}")
        distances, indices = self.index.search(query_vectors, k)
        
        self.logger.debug(f"Search results: distances shape={distances.shape}, indices shape={indices.shape}")
        self.logger.debug(f"Distances: {distances}, Indices: {indices}")
        
        self.logger.debug("Performed search with %d query vectors, k=%d", len(query_vectors), k)
        return distances, indices
    
    def save_index(self, filename: str = 'index.faiss'):
        """Save index to disk."""
        path = os.path.join(self.storage_path, filename)
        faiss.write_index(self.index, path)
        self.logger.info("Saved index to %s", path)
        
    def load_index(self, filename: str = 'index.faiss'):
        """Load index from disk."""
        path = os.path.join(self.storage_path, filename)
        self.index = faiss.read_index(path)
        self.logger.info("Loaded index from %s", path)
        
    def get_stats(self) -> dict:
        """Get index statistics."""
        self.logger.info("Getting shard statistics")
        
        # Get memory usage of the index if possible
        index_memory_mb = 0
        try:
            # Estimate memory usage: each vector is dimension * 4 bytes (float32)
            # plus overhead for the IDs (8 bytes per ID)
            if hasattr(self.index, 'ntotal') and hasattr(self.index, 'd'):
                vector_memory = self.index.ntotal * self.index.d * 4  # float32 = 4 bytes
                id_memory = self.index.ntotal * 8  # int64 = 8 bytes
                index_memory_mb = (vector_memory + id_memory) / (1024 * 1024)  # Convert to MB
        except Exception as e:
            self.logger.warning(f"Failed to estimate memory usage: {str(e)}")
        
        # Get a sample of vector IDs if available
        sample_ids = []
        try:
            if hasattr(self.index, 'id_map') and self.index.ntotal > 0:
                # Get up to 5 IDs as a sample
                sample_size = min(5, self.index.ntotal)
                sample_query = np.zeros((1, self.dimension), dtype=np.float32)
                _, sample_indices = self.index.search(sample_query, sample_size)
                sample_ids = sample_indices[0].tolist()
        except Exception as e:
            self.logger.warning(f"Failed to get sample IDs: {str(e)}")
            
        return {
            'total_vectors': self.index.ntotal if hasattr(self.index, 'ntotal') else 0,
            'dimension': self.index.d if hasattr(self.index, 'd') else self.dimension,
            'index_type': 'FlatL2',
            'metric': 'L2',
            'memory_usage_mb': round(index_memory_mb, 2),
            'next_id': self.next_id,
            'sample_ids': sample_ids,
            'storage_path': self.storage_path,
            'timestamp': time.time()
        }
