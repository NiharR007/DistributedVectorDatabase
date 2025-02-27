#!/usr/bin/env python
"""
Visualize search results from the COCO dataset.
"""

import os
import json
import numpy as np
import requests
import argparse
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm
from sklearn.decomposition import PCA

def load_embeddings(embeddings_file):
    """Load embeddings from NPZ file."""
    data = np.load(embeddings_file)
    embeddings = data['embeddings']
    image_ids = data['image_ids']
    return embeddings, image_ids

def reduce_dimensions(embeddings, target_dim=512, method='pca'):
    """Reduce the dimensionality of embeddings."""
    print(f"Reducing dimensions from {embeddings.shape[1]} to {target_dim} using {method}")
    
    if method == 'pca':
        pca = PCA(n_components=target_dim)
        reduced_embeddings = pca.fit_transform(embeddings)
        explained_variance = sum(pca.explained_variance_ratio_) * 100
        print(f"PCA explained variance: {explained_variance:.2f}%")
    else:
        raise ValueError(f"Unsupported dimension reduction method: {method}")
    
    return reduced_embeddings

def get_image_path(image_id, coco_dir, image_set='val2017'):
    """Get path to COCO image."""
    return os.path.join(coco_dir, image_set, f"{image_id:012d}.jpg")

def search_similar_images(query_vector, k, coordinator_url):
    """Search for similar images."""
    payload = {
        'query_vector': query_vector.tolist(),
        'k': k
    }
    
    try:
        print(f"Sending search request to {coordinator_url}/search")
        response = requests.post(
            f"{coordinator_url}/search",
            json=payload
        )
        
        if response.status_code != 200:
            print(f"Error searching: {response.text}")
            return [], []
        
        result = response.json()
        
        # Check if result contains warning or error
        if 'warning' in result:
            print(f"Warning from server: {result['warning']}")
            
        if 'error' in result:
            print(f"Error from server: {result['error']}")
            return [], []
            
        # Check if distances and indices are present and non-empty
        if 'distances' not in result or 'indices' not in result:
            print(f"Invalid response format: {result}")
            return [], []
            
        distances = result["distances"]
        indices = result["indices"]
        
        # Check if we got any results
        if len(distances) == 0 or len(indices) == 0:
            print("No search results returned")
            return [], []
            
        print(f"Received {len(indices)} search results")
        return distances, indices
    except Exception as e:
        print(f"Error during search: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return [], []

def visualize_search_results(query_id, similar_ids, distances, coco_dir, image_set='val2017', output_file=None):
    """Visualize query image and similar images."""
    n_results = len(similar_ids)
    
    # Handle case when no results are returned
    if n_results == 0:
        print(f"Warning: No similar images found for query ID: {query_id}")
        # Create figure with just the query image
        fig = plt.figure(figsize=(8, 8))
        
        # Plot query image
        query_img_path = get_image_path(query_id, coco_dir, image_set)
        if os.path.exists(query_img_path):
            query_img = Image.open(query_img_path)
            plt.imshow(query_img)
            plt.title(f"Query Image (ID: {query_id})\nNo similar images found")
            plt.axis('off')
        else:
            print(f"Query image not found: {query_img_path}")
            
        if output_file:
            plt.savefig(output_file, bbox_inches='tight')
            print(f"Saved visualization to {output_file}")
        
        plt.close()
        return
    
    # Create figure with grid
    fig = plt.figure(figsize=(15, 8))
    gs = gridspec.GridSpec(2, (n_results + 1) // 2)
    
    # Plot query image
    query_img_path = get_image_path(query_id, coco_dir, image_set)
    if os.path.exists(query_img_path):
        query_img = Image.open(query_img_path)
        ax = plt.subplot(gs[0, 0])
        ax.imshow(query_img)
        ax.set_title(f"Query Image (ID: {query_id})")
        ax.axis('off')
    else:
        print(f"Query image not found: {query_img_path}")
    
    # Plot similar images
    for i, (sim_id, distance) in enumerate(zip(similar_ids, distances)):
        row, col = (i + 1) // ((n_results + 1) // 2), (i + 1) % ((n_results + 1) // 2)
        
        sim_img_path = get_image_path(sim_id, coco_dir, image_set)
        if os.path.exists(sim_img_path):
            sim_img = Image.open(sim_img_path)
            ax = plt.subplot(gs[row, col])
            ax.imshow(sim_img)
            ax.set_title(f"Similar #{i+1} (ID: {sim_id})\nDistance: {distance:.2f}")
            ax.axis('off')
        else:
            print(f"Similar image not found: {sim_img_path}")
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved visualization to {output_file}")
    else:
        plt.show()
    
    plt.close()

def run_multiple_queries(embeddings, image_ids, coordinator_url, coco_dir, 
                         num_queries=5, k=5, image_set='val2017', output_dir='visualizations'):
    """Run multiple queries and visualize results."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Select random query vectors
    query_indices = np.random.choice(len(embeddings), num_queries, replace=False)
    
    for i, idx in enumerate(query_indices):
        query_vector = embeddings[idx]
        query_id = image_ids[idx]
        
        print(f"Query {i+1}/{num_queries} - Image ID: {query_id}")
        
        distances, similar_ids = search_similar_images(query_vector, k, coordinator_url)
        
        # Always visualize, even with empty results
        output_file = os.path.join(output_dir, f"query_{i+1}_id_{query_id}.png")
        visualize_search_results(
            query_id, similar_ids, distances, 
            coco_dir, image_set, output_file
        )

def main(args):
    # Load embeddings
    print(f"Loading embeddings from {args.embeddings}")
    embeddings, image_ids = load_embeddings(args.embeddings)
    print(f"Loaded {len(embeddings)} embeddings with shape {embeddings.shape}")
    
    # Reduce dimensions if needed
    if embeddings.shape[1] != args.target_dim:
        embeddings = reduce_dimensions(embeddings, args.target_dim, args.reduction_method)
        print(f"Reduced embeddings shape: {embeddings.shape}")
    
    coordinator_url = f"http://{args.host}:{args.port}"
    
    if args.query_id is not None:
        # Find the index of the query image
        query_idx = np.where(image_ids == args.query_id)[0]
        if len(query_idx) == 0:
            print(f"Image ID {args.query_id} not found in embeddings")
            return
        
        query_idx = query_idx[0]
        query_vector = embeddings[query_idx]
        
        print(f"Searching for images similar to image ID: {args.query_id}")
        distances, similar_ids = search_similar_images(
            query_vector, args.k, coordinator_url
        )
        
        if distances is not None:
            output_file = f"query_id_{args.query_id}.png" if args.output is None else args.output
            visualize_search_results(
                args.query_id, similar_ids, distances, 
                args.coco_dir, args.image_set, output_file
            )
    else:
        # Run multiple random queries
        run_multiple_queries(
            embeddings, image_ids, coordinator_url, args.coco_dir,
            args.num_queries, args.k, args.image_set, args.output_dir
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize search results from COCO dataset")
    parser.add_argument("--embeddings", type=str, default="data/coco_embeddings.npz", help="Path to embeddings file")
    parser.add_argument("--coco-dir", type=str, default="data/coco", help="Path to COCO dataset directory")
    parser.add_argument("--image-set", type=str, default="val2017", help="Image set (train2017, val2017)")
    parser.add_argument("--host", type=str, default="localhost", help="Coordinator host")
    parser.add_argument("--port", type=int, default=6000, help="Coordinator port")
    parser.add_argument("--query-id", type=int, help="Specific image ID to query")
    parser.add_argument("--k", type=int, default=5, help="Number of similar images to find")
    parser.add_argument("--num-queries", type=int, default=5, help="Number of random queries to run")
    parser.add_argument("--output", type=str, help="Output file for single query visualization")
    parser.add_argument("--output-dir", type=str, default="visualizations", help="Output directory for multiple query visualizations")
    parser.add_argument("--target-dim", type=int, default=512, help="Target dimension for embeddings")
    parser.add_argument("--reduction-method", type=str, default="pca", choices=["pca"], help="Dimension reduction method")
    
    args = parser.parse_args()
    main(args)
