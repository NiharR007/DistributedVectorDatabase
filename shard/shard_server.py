#!/usr/bin/env python
"""
Shard server for the distributed vector database.
This script starts a Flask server that uses the ShardNode class to handle requests.
"""

import os
import sys
import argparse
import logging
import yaml
import numpy as np
from flask import Flask, request, jsonify
from shard_node import ShardNode

app = Flask(__name__)
shard = None
logger = logging.getLogger(__name__)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "healthy"})

@app.route('/add_vectors', methods=['POST'])
def add_vectors():
    """Add vectors to the shard."""
    try:
        data = request.json
        
        if 'vectors' not in data:
            logger.error("No vectors provided")
            return jsonify({"error": "No vectors provided"}), 400
        
        vectors = np.array(data['vectors'], dtype=np.float32)
        logger.info(f"Received request to add {len(vectors)} vectors with shape {vectors.shape}")
        
        if 'ids' in data and data['ids'] is not None:
            ids = np.array(data['ids'], dtype=np.int64)
            logger.debug(f"IDs provided with shape {ids.shape}")
            logger.debug(f"Sample IDs: {ids[:5]}...")
        else:
            ids = None
            logger.debug("No IDs provided, will generate automatically")
        
        try:
            logger.debug("Calling shard.add_vectors")
            shard.add_vectors(vectors, ids)
            
            # Get stats to verify vectors were added
            stats = shard.get_stats()
            logger.info(f"Shard stats after adding vectors: {stats}")
            
            return jsonify({
                "status": "success", 
                "added": len(vectors),
                "total_vectors": stats.get("total_vectors", 0)
            })
        except Exception as e:
            logger.error(f"Error in shard.add_vectors: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return jsonify({"error": str(e)}), 500
    except Exception as e:
        logger.error(f"Error processing add_vectors request: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500

@app.route('/search', methods=['POST'])
def search():
    """Search for similar vectors."""
    data = request.json
    
    if 'query_vector' not in data:
        return jsonify({"error": "No query vector provided"}), 400
    
    if 'k' not in data:
        return jsonify({"error": "No k value provided"}), 400
    
    query_vector = np.array(data['query_vector'], dtype=np.float32)
    
    # Reshape if it's a single vector
    if len(query_vector.shape) == 1:
        query_vector = query_vector.reshape(1, -1)
    
    k = int(data['k'])
    
    logger.debug(f"Received search request with query_vector shape={query_vector.shape}, k={k}")
    
    try:
        logger.debug("Calling shard.search")
        distances, indices = shard.search(query_vector, k)
        
        logger.debug(f"Search results: distances shape={distances.shape}, indices shape={indices.shape}")
        logger.debug(f"Distances: {distances}, Indices: {indices}")
        
        # Convert to Python types for JSON serialization
        distances = distances.tolist()
        indices = indices.tolist()
        
        logger.debug(f"Returning search results: {len(indices)} items")
        
        return jsonify({
            "status": "success",
            "distances": distances,
            "indices": indices
        })
    except Exception as e:
        logger.error(f"Error searching: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500

@app.route('/stats', methods=['GET'])
def get_stats():
    """Get shard statistics."""
    try:
        stats = shard.get_stats()
        return jsonify(stats)
    except Exception as e:
        logger.error("Error getting stats: %s", str(e))
        return jsonify({"error": str(e)}), 500

def setup_logging(log_level='INFO'):
    """Setup logging configuration."""
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")
    
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def main(args):
    """Main function to start the shard server."""
    global shard
    
    # Setup logging
    setup_logging(args.log_level)
    
    # Create config if it doesn't exist
    if not os.path.exists(args.config):
        logger.info("Config file not found, creating default config")
        
        config = {
            "storage_path": args.data_dir,
            "index_type": "HNSW",
            "index_params": {
                "space": "cosine",
                "M": 16,
                "ef_construction": 200,
                "ef_search": 100
            },
            "monitoring": {
                "enable_metrics": True,
                "log_level": args.log_level,
                "performance_tracking": True
            }
        }
        
        os.makedirs(os.path.dirname(args.config), exist_ok=True)
        with open(args.config, 'w') as f:
            yaml.dump(config, f)
    
    # Initialize shard node
    try:
        # Update config with data_dir if provided
        if args.data_dir:
            with open(args.config, 'r') as f:
                config = yaml.safe_load(f)
            
            config['storage_path'] = args.data_dir
            
            with open(args.config, 'w') as f:
                yaml.dump(config, f)
        
        shard = ShardNode(args.config)
        logger.info("Initialized shard node with config: %s", args.config)
    except Exception as e:
        logger.error("Error initializing shard node: %s", str(e))
        sys.exit(1)
    
    # Start Flask server
    logger.info("Starting shard server on %s:%s", args.host, args.port)
    app.run(host=args.host, port=args.port, debug=args.debug)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start a shard server for the distributed vector database")
    parser.add_argument("--host", type=str, default="localhost", help="Host to bind the server to")
    parser.add_argument("--port", type=int, default=6001, help="Port to bind the server to")
    parser.add_argument("--config", type=str, default="config/shard_node.yaml", help="Path to shard node config")
    parser.add_argument("--data-dir", type=str, default="data/shard", help="Directory to store shard data")
    parser.add_argument("--log-level", type=str, default="DEBUG", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="Logging level")
    parser.add_argument("--debug", action="store_true", help="Run Flask in debug mode")
    
    args = parser.parse_args()
    main(args)
