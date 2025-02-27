from flask import Flask, request, jsonify
import numpy as np
import yaml
from coordinator.coordinator import Coordinator
import logging

app = Flask(__name__)
coordinator = None

# Initialize coordinator at startup
def init_coordinator():
    global coordinator
    with open('config/coordinator.yaml', 'r') as f:
        config = yaml.safe_load(f)
    coordinator = Coordinator(config)

@app.route('/add_vectors', methods=['POST'])
def add_vectors():
    """Add vectors to the distributed system."""
    data = request.get_json()
    vectors = np.array(data['vectors'])
    ids = np.array(data['ids']) if 'ids' in data else None
    
    app.logger.info(f"Coordinator received request to add {len(vectors)} vectors with shape {vectors.shape}")
    if ids is not None:
        app.logger.info(f"Vector IDs: {ids}")
    
    try:
        coordinator.distribute_vectors(vectors, ids)
        app.logger.info("Successfully distributed vectors to shards")
        return jsonify({'status': 'success', 'message': f'Added {len(vectors)} vectors'}), 200
    except Exception as e:
        app.logger.error(f"Error distributing vectors: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 400
        
@app.route('/search', methods=['POST'])
def search():
    """Perform similarity search."""
    data = request.get_json()
    query_vector = np.array(data['query_vector'])
    k = int(data['k'])
    
    try:
        distances, indices = coordinator.search(query_vector, k)
        return jsonify({
            'status': 'success',
            'distances': distances.tolist(),
            'indices': indices.tolist()
        }), 200
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400
        
@app.route('/stats', methods=['GET'])
def get_stats():
    """Get system statistics."""
    try:
        stats = coordinator.get_system_stats()
        return jsonify(stats), 200
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400
        
if __name__ == '__main__':
    init_coordinator()
    logging.basicConfig(level=logging.INFO)
    app.run(host='0.0.0.0', port=6000)
