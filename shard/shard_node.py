import faiss
import numpy as np
import yaml
import logging
from typing import List, Tuple, Optional, Union
import os

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
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
            
        if vectors.dtype != np.float32:
            vectors = vectors.astype(np.float32)
            
        n_vectors = vectors.shape[0]
        
        if ids is None:
            ids = np.arange(self.next_id, self.next_id + n_vectors)
            self.next_id += n_vectors
            
        logging.info(f"Adding {n_vectors} vectors with IDs: {ids}")
        self.index.add_with_ids(vectors, ids)
            
        self.logger.info("Added %d vectors to index", len(vectors))
        
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
        return {
            'total_vectors': self.index.ntotal,
            'dimension': self.index.d,
            'index_type': 'FlatL2',
            'metric': 'L2'
        }
