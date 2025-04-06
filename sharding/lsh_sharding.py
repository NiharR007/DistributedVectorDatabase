import numpy as np
from sklearn.random_projection import GaussianRandomProjection
from typing import List, Tuple, Set
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
        dummy_data = np.random.randn(max(num_hash_functions * 2, 100), input_dim)
        for projection in self.random_projections:
            projection.fit(dummy_data)
        
        logging.info(f"Initialized LSH sharding with {num_hash_tables} tables and {num_hash_functions} hash functions")
    
    def _preprocess_vector(self, vector: np.ndarray) -> np.ndarray:
        """Preprocess vector for LSH hashing.
        
        Args:
            vector: Input vector
            
        Returns:
            Preprocessed vector
        """
        # Ensure vector is 2D
        if vector.ndim == 1:
            vector = vector.reshape(1, -1)
            
        # Convert to float32 for LSH
        if vector.dtype != np.float32:
            vector = vector.astype(np.float32)
            
        # Normalize vector for better LSH performance
        norms = np.linalg.norm(vector, axis=1, keepdims=True)
        # Avoid division by zero
        norms[norms == 0] = 1.0
        normalized_vector = vector / norms
        
        return normalized_vector
    
    def _compute_hash_values(self, vector: np.ndarray) -> List[np.ndarray]:
        """Compute LSH hash values for a vector.
        
        Args:
            vector: Input vector (preprocessed)
            
        Returns:
            List of hash values from each hash table
        """
        hash_values = []
        for i, projection in enumerate(self.random_projections):
            projected = projection.transform(vector)
            # Convert continuous projections to binary
            hash_value = (projected > 0).astype(int)
            hash_values.append(hash_value)
            logging.debug(f"Table {i} hash value shape: {hash_value.shape}")
            
        return hash_values
    
    def get_shard_id(self, vector: np.ndarray) -> int:
        """Determine shard ID for a given vector using LSH.
        
        Args:
            vector: Input vector to be sharded
            
        Returns:
            Shard ID as an integer
        """
        vector = self._preprocess_vector(vector)
        hash_values = self._compute_hash_values(vector)
            
        # Combine hash values from all tables
        combined_hash = np.concatenate(hash_values, axis=1)
        
        # Use weighted sum for more consistent hashing
        weights = 2 ** np.arange(combined_hash.shape[1])
        shard_id = int(np.sum(combined_hash * weights) % self.num_hash_tables)
        
        logging.debug(f"Assigned vector to shard {shard_id}")
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
        
        # Preprocess query vector
        query_vector = self._preprocess_vector(query_vector)
        
        # First, get the primary shard for this vector
        primary_shard = self.get_shard_id(query_vector)
        candidates = {primary_shard}
        
        # For each hash table, compute hash and find nearby shards
        for i, projection in enumerate(self.random_projections):
            projected = projection.transform(query_vector)
            
            # Find points close to decision boundaries
            distances = np.abs(projected)
            closest_indices = np.argsort(distances.flatten())[:min(2, len(distances.flatten()))]
            
            # Flip bits at these positions to get neighboring buckets
            for idx in closest_indices:
                table_idx = idx // self.num_hash_functions
                bit_idx = idx % self.num_hash_functions
                
                # Create a perturbed hash by flipping a bit
                hash_values = self._compute_hash_values(query_vector)
                perturbed_hash = hash_values.copy()
                
                # Flip the bit at the decision boundary
                bit_to_flip = perturbed_hash[table_idx][0, bit_idx]
                perturbed_hash[table_idx][0, bit_idx] = 1 - bit_to_flip
                
                # Compute shard ID for this perturbed hash
                combined_hash = np.concatenate(perturbed_hash, axis=1)
                weights = 2 ** np.arange(combined_hash.shape[1])
                perturbed_shard = int(np.sum(combined_hash * weights) % self.num_hash_tables)
                
                candidates.add(perturbed_shard)
                
                if len(candidates) >= num_candidates:
                    break
            
            if len(candidates) >= num_candidates:
                break
        
        # If we still don't have enough candidates, add more shards sequentially
        if len(candidates) < num_candidates:
            for i in range(self.num_hash_tables):
                if i not in candidates:
                    candidates.add(i)
                if len(candidates) >= num_candidates:
                    break
        
        candidate_list = list(candidates)[:num_candidates]
        logging.debug(f"Selected candidate shards: {candidate_list}")
        return candidate_list
    
    def batch_get_shard_ids(self, vectors: np.ndarray) -> np.ndarray:
        """Get shard IDs for multiple vectors.
        
        Args:
            vectors: Input vectors (n_vectors, dim)
            
        Returns:
            Array of shard IDs
        """
        logging.info(f"LSH: Processing {len(vectors)} vectors with shape {vectors.shape}")
        
        # Preprocess vectors
        vectors = self._preprocess_vector(vectors)
        
        # Get LSH hash values from all tables
        hash_values = self._compute_hash_values(vectors)
            
        # Combine hash values from all tables
        combined_hash = np.concatenate(hash_values, axis=1)
        logging.info(f"LSH: Combined hash shape: {combined_hash.shape}")
        
        # Use weighted sum for more consistent hashing
        weights = 2 ** np.arange(combined_hash.shape[1])
        shard_ids = np.sum(combined_hash * weights, axis=1) % self.num_hash_tables
        
        unique_shards, counts = np.unique(shard_ids, return_counts=True)
        logging.info(f"LSH: Assigned vectors to shards: {list(zip(unique_shards, counts))}")
        return shard_ids
