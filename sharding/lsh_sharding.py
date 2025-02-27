import numpy as np
from sklearn.random_projection import GaussianRandomProjection
from typing import List, Tuple
import logging

class LSHSharding:
    def __init__(self, num_hash_functions: int, num_hash_tables: int, input_dim: int):
        """Initialize LSH-based sharding strategy.
        
        Args:
            num_hash_functions: Number of hash functions per table
            num_hash_tables: Number of hash tables
            input_dim: Dimension of input vectors
        """
        self.num_hash_functions = num_hash_functions
        self.num_hash_tables = num_hash_tables
        self.input_dim = input_dim
        
        # Initialize random projections for LSH
        self.random_projections = [
            GaussianRandomProjection(n_components=num_hash_functions)
            for _ in range(num_hash_tables)
        ]
        
        # Fit random projections with dummy data
        dummy_data = np.random.randn(num_hash_functions * 2, input_dim)
        for projection in self.random_projections:
            projection.fit(dummy_data)
    
    def get_shard_id(self, vector: np.ndarray) -> int:
        """Determine shard ID for a given vector using LSH.
        
        Args:
            vector: Input vector to be sharded
            
        Returns:
            Shard ID as an integer
        """
        if vector.ndim == 1:
            vector = vector.reshape(1, -1)
            
        # Get LSH hash values from all tables
        hash_values = []
        for projection in self.random_projections:
            projected = projection.transform(vector)
            # Convert continuous projections to binary
            hash_value = (projected > 0).astype(int)
            hash_values.append(hash_value)
            
        # Combine hash values from all tables
        combined_hash = np.concatenate(hash_values, axis=1)
        # Convert binary hash to integer for shard assignment
        shard_id = int(np.sum(combined_hash) % self.num_hash_tables)
        
        return shard_id
    
    def get_candidate_shards(self, query_vector: np.ndarray, num_candidates: int = 2) -> List[int]:
        """Get candidate shards for query vector.
        
        Args:
            query_vector: Query vector
            num_candidates: Number of candidate shards to return
            
        Returns:
            List of candidate shard IDs
        """
        logging.debug(f"Getting candidate shards for query vector with shape {query_vector.shape}")
        logging.debug(f"num_hash_tables: {self.num_hash_tables}")
        
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)
            
        # Get hash values and distances for all tables
        shard_distances = []
        for i, projection in enumerate(self.random_projections):
            projected = projection.transform(query_vector)
            # Calculate distance to shard boundaries
            distances = np.abs(projected)
            logging.debug(f"Table {i} projected shape: {projected.shape}, distances shape: {distances.shape}")
            shard_distances.extend(distances.flatten())
            
        logging.debug(f"shard_distances: {shard_distances}")
        
        # Sort shards by distance and return top candidates
        num_candidates = min(num_candidates, len(shard_distances))
        logging.debug(f"Using num_candidates: {num_candidates}")
        
        if num_candidates == 0:
            logging.warning("No candidate shards found")
            # Return all shards if no candidates found
            return list(range(self.num_hash_tables))
            
        candidate_indices = np.argsort(shard_distances)[:num_candidates]
        logging.debug(f"candidate_indices: {candidate_indices}")
        
        candidate_shards = [int(idx % self.num_hash_tables) for idx in candidate_indices]
        logging.debug(f"candidate_shards: {candidate_shards}")
        
        return candidate_shards
    
    def batch_get_shard_ids(self, vectors: np.ndarray) -> np.ndarray:
        """Get shard IDs for multiple vectors.
        
        Args:
            vectors: Input vectors (n_vectors, dim)
            
        Returns:
            Array of shard IDs
        """
        logging.info(f"LSH: Processing {len(vectors)} vectors with shape {vectors.shape}")
        
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
            
        # Convert to float32 for LSH
        if vectors.dtype != np.float32:
            vectors = vectors.astype(np.float32)
            
        # Normalize vectors for better LSH performance
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        normalized_vectors = vectors / norms
        
        # Get LSH hash values from all tables
        hash_values = []
        for i, projection in enumerate(self.random_projections):
            projected = projection.transform(normalized_vectors)
            # Convert continuous projections to binary
            hash_value = (projected > 0).astype(int)
            hash_values.append(hash_value)
            logging.info(f"LSH: Table {i} hash values shape: {hash_value.shape}")
            
        # Combine hash values from all tables
        combined_hash = np.concatenate(hash_values, axis=1)
        logging.info(f"LSH: Combined hash shape: {combined_hash.shape}")
        
        # Convert binary hash to integer for shard assignment
        # Use modulo to ensure even distribution
        shard_ids = np.sum(combined_hash * (2 ** np.arange(combined_hash.shape[1])), axis=1) % self.num_hash_tables
        
        unique_shards, counts = np.unique(shard_ids, return_counts=True)
        logging.info(f"LSH: Assigned vectors to shards: {list(zip(unique_shards, counts))}")
        return shard_ids
