import yaml
import numpy as np
import requests
import logging
from typing import List, Dict, Tuple, Union
import os
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
import time
import sys

# Add parent directory to path to import from sharding
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sharding.lsh_sharding import LSHSharding

class Coordinator:
    def __init__(self, config: Union[str, dict]):
        """Initialize coordinator node.
        
        Args:
            config: Either a path to coordinator configuration file or a config dictionary
        """
        self.load_config(config)
        self.setup_logging()
        self.initialize_sharding()
        
    def load_config(self, config: Union[str, dict]):
        """Load configuration from yaml file or dict."""
        if isinstance(config, str):
            with open(config, 'r') as f:
                config = yaml.safe_load(f)
                
        self.nodes = config['nodes']
        self.sharding_strategy = config['sharding_strategy']
        self.replication_factor = config['replication_factor']
        self.query_timeout = config['query_timeout']
        self.consistency_mode = config['consistency_mode']
        self.lsh_config = config['lsh_config']
        
    def setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def initialize_sharding(self):
        """Initialize sharding strategy."""
        if self.sharding_strategy == 'lsh':
            self.sharding = LSHSharding(
                num_hash_functions=self.lsh_config['num_hash_functions'],
                num_hash_tables=self.lsh_config['num_hash_tables'],
                input_dim=self.lsh_config['input_dim']
            )
        else:
            raise ValueError(f"Unsupported sharding strategy: {self.sharding_strategy}")
            
    def _get_node_url(self, node: Dict[str, str]) -> str:
        """Get URL for a node."""
        return f"http://{node['host']}:{node['port']}"
    
    def distribute_vectors(self, vectors: np.ndarray, ids: np.ndarray = None):
        """Distribute vectors across shard nodes.
        
        Args:
            vectors: Vectors to distribute (n_vectors, dim)
            ids: Optional vector IDs
        """
        self.logger.info(f"Starting vector distribution for {len(vectors)} vectors")
        
        # Get shard assignments for all vectors
        shard_assignments = self.sharding.batch_get_shard_ids(vectors)
        self.logger.info(f"Shard assignments: {np.unique(shard_assignments, return_counts=True)}")
        
        # Group vectors by shard
        shard_vectors = {}
        shard_ids = {}
        for i, shard_id in enumerate(shard_assignments):
            if shard_id not in shard_vectors:
                shard_vectors[shard_id] = []
                shard_ids[shard_id] = [] if ids is not None else None
            shard_vectors[shard_id].append(vectors[i])
            if ids is not None:
                shard_ids[shard_id].append(ids[i])
        
        self.logger.info(f"Grouped vectors by shard: {[f'shard_{k}: {len(v)} vectors' for k, v in shard_vectors.items()]}")
        
        # Send vectors to appropriate shards
        with ThreadPoolExecutor(max_workers=len(self.nodes)) as executor:
            futures = []
            for shard_id, vectors in shard_vectors.items():
                node = self.nodes[shard_id % len(self.nodes)]
                vectors_array = np.array(vectors)
                ids_array = np.array(shard_ids[shard_id]) if shard_ids[shard_id] is not None else None
                
                self.logger.info(f"Sending {len(vectors_array)} vectors to node {node['host']}:{node['port']}")
                futures.append(
                    executor.submit(
                        self._send_vectors_to_shard,
                        node,
                        vectors_array,
                        ids_array
                    )
                )
            
            # Wait for all transfers to complete
            concurrent.futures.wait(futures)
            self.logger.info("Completed vector distribution")
            
    def _send_vectors_to_shard(self, node: Dict[str, str], vectors: np.ndarray, ids: np.ndarray = None):
        """Send vectors to a shard node.
        
        Args:
            node: Node configuration
            vectors: Vectors to send
            ids: Optional vector IDs
        """
        url = f"{self._get_node_url(node)}/add_vectors"
        data = {
            'vectors': vectors.tolist(),
            'ids': ids.tolist() if ids is not None else None
        }
        
        try:
            response = requests.post(url, json=data)
            response.raise_for_status()
            self.logger.info(f"Successfully sent {len(vectors)} vectors to node {node['host']}:{node['port']}")
        except Exception as e:
            self.logger.error(f"Failed to send vectors to node {node['host']}:{node['port']}: {str(e)}")
            
    def search(self, query_vector: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Perform distributed similarity search.
        
        Args:
            query_vector: Query vector
            k: Number of nearest neighbors to return
            
        Returns:
            Tuple of (distances, indices)
        """
        self.logger.debug(f"Starting search with query_vector shape={query_vector.shape}, k={k}")
        
        # Ensure k is at least 1
        if k <= 0:
            self.logger.warning(f"Invalid k value: {k}, setting to 1")
            k = 1
        
        # Get candidate shards for query
        candidate_shards = self.sharding.get_candidate_shards(query_vector)
        self.logger.debug(f"Candidate shards: {candidate_shards}")
        
        # If no candidate shards, return empty results
        if not candidate_shards:
            self.logger.warning("No candidate shards found for query")
            return np.array([]), np.array([])
        
        # Search in parallel across candidate shards
        with ThreadPoolExecutor(max_workers=len(candidate_shards)) as executor:
            futures = []
            for shard_id in candidate_shards:
                # Ensure shard_id is within range of available nodes
                node_idx = shard_id % len(self.nodes)
                self.logger.debug(f"Mapping shard_id {shard_id} to node_idx {node_idx} (out of {len(self.nodes)} nodes)")
                
                node = self.nodes[node_idx]
                self.logger.debug(f"Submitting search to shard {shard_id} at {node['host']}:{node['port']}")
                futures.append(
                    executor.submit(
                        self._search_shard,
                        node,
                        query_vector,
                        k
                    )
                )
                
            # Collect and merge results
            all_distances = []
            all_indices = []
            for future in concurrent.futures.as_completed(futures):
                try:
                    distances, indices = future.result(timeout=self.query_timeout)
                    self.logger.debug(f"Received search results: distances shape={distances.shape if hasattr(distances, 'shape') else 'None'}, indices shape={indices.shape if hasattr(indices, 'shape') else 'None'}")
                    
                    # Skip empty results
                    if len(distances) == 0 or len(indices) == 0:
                        self.logger.debug("Skipping empty result")
                        continue
                        
                    # Check if distances and indices have the same shape
                    if len(distances) != len(indices):
                        self.logger.warning(f"Mismatched shapes: distances={len(distances)}, indices={len(indices)}")
                        continue
                    
                    all_distances.extend(distances)
                    all_indices.extend(indices)
                except Exception as e:
                    self.logger.error(f"Search failed on a shard: {str(e)}")
                    import traceback
                    self.logger.error(f"Traceback: {traceback.format_exc()}")
                    
        # Sort and return top k results
        if not all_distances:
            self.logger.warning("No search results found across all shards")
            return np.array([]), np.array([])
            
        self.logger.debug(f"All distances: {all_distances}")
        self.logger.debug(f"All indices: {all_indices}")
            
        try:
            all_distances = np.array(all_distances)
            all_indices = np.array(all_indices)
            
            self.logger.debug(f"all_distances shape: {all_distances.shape}, all_indices shape: {all_indices.shape}")
            
            # Sort by distance
            k = min(k, len(all_distances))  # Ensure k is not larger than the number of results
            if k == 0:
                self.logger.warning("k is 0 after adjustment")
                return np.array([]), np.array([])
                
            self.logger.debug(f"Using k={k} for sorting")
            
            # Check if we have enough results to sort
            if len(all_distances) < k:
                self.logger.warning(f"Not enough results: {len(all_distances)} < {k}")
                k = len(all_distances)
                
            sorted_idx = np.argsort(all_distances)[:k]
            self.logger.debug(f"sorted_idx: {sorted_idx}")
            
            # Check if sorted_idx is valid
            if len(sorted_idx) == 0:
                self.logger.warning("No valid indices after sorting")
                return np.array([]), np.array([])
                
            # Check if sorted_idx is within bounds
            if np.max(sorted_idx) >= len(all_distances):
                self.logger.error(f"Index out of bounds: max sorted_idx={np.max(sorted_idx)}, len(all_distances)={len(all_distances)}")
                # Clip indices to valid range
                sorted_idx = sorted_idx[sorted_idx < len(all_distances)]
                if len(sorted_idx) == 0:
                    return np.array([]), np.array([])
            
            result_distances = all_distances[sorted_idx]
            result_indices = all_indices[sorted_idx]
            
            self.logger.debug(f"Returning results: distances={result_distances}, indices={result_indices}")
            return result_distances, result_indices
        except Exception as e:
            self.logger.error(f"Error processing search results: {str(e)}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return np.array([]), np.array([])
            
    def _search_shard(self, node: Dict[str, str], query_vector: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Search on a specific shard node.
        
        Args:
            node: Node configuration
            query_vector: Query vector
            k: Number of nearest neighbors
            
        Returns:
            Tuple of (distances, indices)
        """
        url = f"{self._get_node_url(node)}/search"
        data = {
            'query_vector': query_vector.tolist(),
            'k': k
        }
        
        try:
            self.logger.debug(f"Sending search request to {url}")
            response = requests.post(url, json=data, timeout=self.query_timeout)
            response.raise_for_status()
            result = response.json()
            self.logger.debug(f"Received response from {url}: {result}")
            
            if 'error' in result:
                self.logger.error(f"Error from {url}: {result['error']}")
                return np.array([]), np.array([])
            
            if 'distances' not in result or 'indices' not in result:
                self.logger.error(f"Invalid response from {url}: {result}")
                return np.array([]), np.array([])
            
            # Validate the response data
            distances = np.array(result['distances'])
            indices = np.array(result['indices'])
            
            # Check if distances and indices have the same shape
            if distances.shape != indices.shape:
                self.logger.error(f"Mismatched shapes from {url}: distances={distances.shape}, indices={indices.shape}")
                return np.array([]), np.array([])
                
            # Check if distances and indices are empty
            if distances.size == 0 or indices.size == 0:
                self.logger.warning(f"Empty results from {url}")
                return np.array([]), np.array([])
                
            self.logger.debug(f"Valid results from {url}: distances shape={distances.shape}, indices shape={indices.shape}")
            return distances, indices
        except requests.exceptions.Timeout:
            self.logger.error(f"Request to {url} timed out after {self.query_timeout} seconds")
            return np.array([]), np.array([])
        except requests.exceptions.ConnectionError:
            self.logger.error(f"Connection error to {url}")
            return np.array([]), np.array([])
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Request error to {url}: {str(e)}")
            return np.array([]), np.array([])
        except ValueError as e:
            self.logger.error(f"JSON parsing error from {url}: {str(e)}")
            return np.array([]), np.array([])
        except Exception as e:
            self.logger.error(f"Search failed on node {node['host']}:{node['port']}: {str(e)}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return np.array([]), np.array([])
            
    def get_system_stats(self) -> Dict:
        """Get statistics from all nodes."""
        stats = {
            'total_nodes': len(self.nodes),
            'sharding_strategy': self.sharding_strategy,
            'replication_factor': self.replication_factor,
            'nodes': {}
        }
        
        for node in self.nodes:
            try:
                url = f"{self._get_node_url(node)}/stats"
                response = requests.get(url)
                response.raise_for_status()
                stats['nodes'][f"{node['host']}:{node['port']}"] = response.json()
            except Exception as e:
                self.logger.error(f"Failed to get stats from node {node['host']}:{node['port']}: {str(e)}")
                
        return stats
