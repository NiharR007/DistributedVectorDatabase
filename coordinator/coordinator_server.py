#!/usr/bin/env python
"""
Coordinator server for the distributed vector database.
This script starts a Flask server that uses the Coordinator class to handle requests.
"""

import os
import sys
import argparse
import logging
import yaml
import numpy as np
from flask import Flask, request, jsonify
from coordinator import Coordinator

app = Flask(__name__)
coordinator = None
logger = logging.getLogger(__name__)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "healthy"})

@app.route('/add_vectors', methods=['POST'])
def add_vectors():
    """Add vectors to the distributed database."""
    try:
        data = request.json
        
        if 'vectors' not in data:
            logger.error("No vectors provided")
            return jsonify({"error": "No vectors provided"}), 400
        
        vectors = np.array(data['vectors'], dtype=np.float32)
        ids = np.array(data['ids']) if 'ids' in data and data['ids'] is not None else None
        
        logger.info(f"Received request to add {len(vectors)} vectors")
        logger.debug(f"Vectors shape: {vectors.shape}")
        if ids is not None:
            logger.debug(f"IDs shape: {ids.shape}")
            logger.debug(f"Sample IDs: {ids[:5]}...")
        
        try:
            logger.debug("Calling coordinator.distribute_vectors")
            coordinator.distribute_vectors(vectors, ids)
            
            # Get system stats to verify vectors were added
            stats = coordinator.get_system_stats()
            logger.info(f"System stats after adding vectors: {stats}")
            
            # Check if vectors were actually added
            total_vectors = 0
            for node_key, node_stats in stats['nodes'].items():
                if 'vector_count' in node_stats:
                    total_vectors += node_stats['vector_count']
            
            logger.info(f"Total vectors in system: {total_vectors}")
            
            return jsonify({
                "status": "success",
                "message": f"Added {len(vectors)} vectors",
                "total_vectors": total_vectors
            })
        except Exception as e:
            logger.error(f"Error distributing vectors: {str(e)}")
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
    try:
        data = request.json
        
        if 'query_vector' not in data:
            logger.error("No query vector provided")
            return jsonify({"error": "No query vector provided"}), 400
        
        if 'k' not in data:
            logger.error("No k value provided")
            return jsonify({"error": "No k value provided"}), 400
        
        query_vector = np.array(data['query_vector'], dtype=np.float32)
        
        # Reshape if it's a single vector
        if len(query_vector.shape) == 1:
            query_vector = query_vector.reshape(1, -1)
        
        k = int(data['k'])
        
        logger.debug(f"Received search request with query_vector shape={query_vector.shape}, k={k}")
        
        # Validate k
        if k <= 0:
            logger.warning(f"Invalid k value: {k}, setting to 1")
            k = 1
        
        try:
            logger.debug("Calling coordinator.search")
            distances, indices = coordinator.search(query_vector, k)
            
            logger.debug(f"Search results: distances shape={distances.shape if hasattr(distances, 'shape') else 'None'}, indices shape={indices.shape if hasattr(indices, 'shape') else 'None'}")
            
            # Check if distances and indices are empty
            if distances.size == 0 or indices.size == 0:
                logger.warning("Empty search results")
                return jsonify({
                    "status": "success",
                    "distances": [],
                    "indices": [],
                    "warning": "No results found"
                })
            
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
            logger.error(f"Error in coordinator.search: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return jsonify({"error": str(e)}), 500
    except Exception as e:
        logger.error(f"Error processing search request: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500

@app.route('/stats', methods=['GET'])
def get_stats():
    """Get system statistics."""
    try:
        logger.info("Getting system statistics")
        stats = coordinator.get_system_stats()
        return jsonify(stats)
    except Exception as e:
        logger.error(f"Error getting system statistics: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
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
    """Main function to start the coordinator server."""
    global coordinator
    
    # Setup logging
    setup_logging(args.log_level)
    
    # Create config if it doesn't exist
    if not os.path.exists(args.config):
        logger.info("Config file not found, creating default config")
        
        config = {
            "host": args.host,
            "port": args.port,
            "sharding_strategy": "lsh",
            "replication_factor": 1,
            "consistency_mode": "eventual",
            "nodes": [
                {"host": "localhost", "port": 6001},
                {"host": "localhost", "port": 6002},
                {"host": "localhost", "port": 6003}
            ]
        }
        
        os.makedirs(os.path.dirname(args.config), exist_ok=True)
        with open(args.config, 'w') as f:
            yaml.dump(config, f)
    
    # Initialize coordinator
    try:
        coordinator = Coordinator(args.config)
        logger.info("Initialized coordinator with config: %s", args.config)
    except Exception as e:
        logger.error("Error initializing coordinator: %s", str(e))
        sys.exit(1)
    
    # Start Flask server
    logger.info("Starting coordinator server on %s:%s", args.host, args.port)
    app.run(host=args.host, port=args.port, debug=args.debug)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start a coordinator server for the distributed vector database")
    parser.add_argument("--host", type=str, default="localhost", help="Host to bind the server to")
    parser.add_argument("--port", type=int, default=6000, help="Port to bind the server to")
    parser.add_argument("--config", type=str, default="config/coordinator.yaml", help="Path to coordinator config")
    parser.add_argument("--log-level", type=str, default="DEBUG", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="Logging level")
    parser.add_argument("--debug", action="store_true", help="Run Flask in debug mode")
    
    args = parser.parse_args()
    main(args)
