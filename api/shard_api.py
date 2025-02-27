from flask import Flask, request, jsonify
import numpy as np
import yaml
from shard.shard_node import ShardNode
import logging

app = Flask(__name__)
shard = None

def init_shard():
    global shard
    with open('config/shard_node.yaml', 'r') as f:
        config = yaml.safe_load(f)
    shard = ShardNode(config)

@app.route('/add_vectors', methods=['POST'])
def add_vectors():
    """Add vectors to the shard."""
    data = request.get_json()
    vectors = np.array(data['vectors'])
    ids = np.array(data['ids']) if data.get('ids') is not None else None
    
    app.logger.info(f"Received request to add {len(vectors)} vectors with shape {vectors.shape}")
    if ids is not None:
        app.logger.info(f"Vector IDs: {ids}")
    
    try:
        # Log index stats before adding vectors
        before_stats = shard.get_stats()
        app.logger.info(f"Index stats before adding vectors: {before_stats}")
        
        shard.add_vectors(vectors, ids)
        
        # Log index stats after adding vectors
        after_stats = shard.get_stats()
        app.logger.info(f"Index stats after adding vectors: {after_stats}")
        
        app.logger.info(f"Successfully added {len(vectors)} vectors. Total vectors in index: {after_stats['total_vectors']}")
        return jsonify({'status': 'success', 'message': f'Added {len(vectors)} vectors'}), 200
    except Exception as e:
        app.logger.error(f"Error adding vectors: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 400

@app.route('/search', methods=['POST'])
def search():
    """Perform similarity search on this shard."""
    data = request.get_json()
    query_vector = np.array(data['query_vector']).reshape(1, -1)
    k = int(data['k'])
    
    try:
        distances, indices = shard.search(query_vector, k)
        return jsonify({
            'status': 'success',
            'distances': distances[0].tolist(),
            'indices': indices[0].tolist()
        }), 200
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

@app.route('/stats', methods=['GET'])
def get_stats():
    """Get shard statistics."""
    try:
        stats = shard.get_stats()
        return jsonify(stats), 200
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

if __name__ == '__main__':
    import sys
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 6001
    init_shard()
    logging.basicConfig(level=logging.INFO)
    app.run(host='0.0.0.0', port=port)
