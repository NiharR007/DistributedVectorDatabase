#!/usr/bin/env python
"""
Load COCO embeddings and add them to the distributed vector database.
"""

import os
import json
import numpy as np
import requests
import argparse
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
from PIL import Image
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

def add_vectors_to_db(vectors, ids, coordinator_url, batch_size=1000):
    """Add vectors to the distributed database in batches."""
    total_vectors = len(vectors)
    print(f"Adding {total_vectors} vectors to the database in batches of {batch_size}")
    print(f"Vector shape: {vectors.shape}, dtype: {vectors.dtype}")
    print(f"IDs shape: {ids.shape}, dtype: {ids.dtype}")
    print(f"Sample vector: {vectors[0][:5]}...")
    print(f"Sample IDs: {ids[:5]}...")
    
    # Verify coordinator is up
    try:
        health_response = requests.get(f"{coordinator_url}/health")
        print(f"Coordinator health check: {health_response.status_code} - {health_response.text}")
    except Exception as e:
        print(f"Coordinator health check failed: {str(e)}")
        return False
    
    batches = (total_vectors + batch_size - 1) // batch_size
    successful_batches = 0
    
    for i in tqdm(range(batches), desc="Adding vector batches"):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, total_vectors)
        
        batch_vectors = vectors[start_idx:end_idx]
        batch_ids = ids[start_idx:end_idx]
        
        print(f"Batch {i+1}/{batches}: Adding {len(batch_vectors)} vectors (IDs {batch_ids[0]}...{batch_ids[-1]})")
        print(f"Batch vectors shape: {batch_vectors.shape}, dtype: {batch_vectors.dtype}")
        
        payload = {
            "vectors": batch_vectors.tolist(),
            "ids": batch_ids.tolist()
        }
        
        try:
            print(f"Sending request to {coordinator_url}/add_vectors")
            print(f"Payload size: {len(str(payload))} characters")
            
            start_time = time.time()
            response = requests.post(
                f"{coordinator_url}/add_vectors",
                json=payload,
                timeout=60  # Add timeout to prevent hanging
            )
            elapsed_time = time.time() - start_time
            
            print(f"Request took {elapsed_time:.2f} seconds")
            print(f"Response status code: {response.status_code}")
            
            if response.status_code != 200:
                print(f"Error adding batch {i+1}: {response.text}")
                continue  # Try next batch instead of failing completely
            
            result = response.json()
            print(f"Response: {result}")
            successful_batches += 1
            
            # Verify vectors were added by checking system stats
            try:
                stats_response = requests.get(f"{coordinator_url}/stats")
                if stats_response.status_code == 200:
                    stats = stats_response.json()
                    print(f"System stats after batch {i+1}: {stats}")
            except Exception as e:
                print(f"Failed to get system stats: {str(e)}")
                
        except Exception as e:
            print(f"Exception adding batch {i+1}: {str(e)}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            # Continue with next batch instead of failing completely
        
        # Add a small delay between batches to avoid overwhelming the system
        time.sleep(1)
    
    print(f"Successfully added {successful_batches}/{batches} batches ({successful_batches*batch_size} vectors) to the database")
    return successful_batches > 0

def search_similar_images(query_vector, k, coordinator_url):
    """Search for similar images."""
    payload = {
        "query_vector": query_vector.tolist(),
        "k": k
    }
    
    response = requests.post(
        f"{coordinator_url}/search",
        json=payload
    )
    
    if response.status_code != 200:
        print(f"Error searching: {response.text}")
        return None, None
    
    result = response.json()
    print(f"Raw search response: {result}")
    
    # Handle different response formats
    if "distances" in result and "indices" in result:
        distances = result["distances"]
        indices = result["indices"]
        
        # Flatten the results if they're nested lists
        flat_distances = []
        flat_indices = []
        
        if isinstance(distances, list):
            for shard_distances in distances:
                if isinstance(shard_distances, list):
                    flat_distances.extend(shard_distances)
                else:
                    flat_distances.append(shard_distances)
                    
        if isinstance(indices, list):
            for shard_indices in indices:
                if isinstance(shard_indices, list):
                    flat_indices.extend(shard_indices)
                else:
                    flat_indices.append(shard_indices)
        
        # Sort results by distance
        if flat_distances and flat_indices:
            sorted_results = sorted(zip(flat_distances, flat_indices))
            sorted_distances, sorted_indices = zip(*sorted_results)
            return sorted_distances[:k], sorted_indices[:k]
        
        return flat_distances, flat_indices
    else:
        # If the response has a different format, try to extract the relevant information
        return result.get("distances", []), result.get("indices", [])

def benchmark_search(embeddings, coordinator_url, num_queries=10, k_values=[1, 5, 10, 50, 100]):
    """Benchmark search performance."""
    # Select random query vectors
    query_indices = np.random.choice(len(embeddings), num_queries, replace=False)
    query_vectors = embeddings[query_indices]
    
    results = {}
    
    for k in k_values:
        latencies = []
        
        for i, query_vector in enumerate(query_vectors):
            start_time = time.time()
            _, _ = search_similar_images(query_vector, k, coordinator_url)
            end_time = time.time()
            
            latency = (end_time - start_time) * 1000  # Convert to ms
            latencies.append(latency)
            
            print(f"Query {i+1}/{num_queries}, k={k}: {latency:.2f} ms")
        
        avg_latency = np.mean(latencies)
        results[k] = avg_latency
        print(f"Average latency for k={k}: {avg_latency:.2f} ms")
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(list(results.keys()), list(results.values()), marker='o')
    plt.xlabel('k (number of results)')
    plt.ylabel('Latency (ms)')
    plt.title('Search Latency vs. k')
    plt.grid(True)
    plt.savefig('search_latency_benchmark.png')
    plt.close()
    
    return results

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
    
    if args.add:
        # Add vectors to the database
        print(f"Adding vectors to the database at {coordinator_url}")
        success = add_vectors_to_db(
            embeddings, image_ids, coordinator_url, args.batch_size
        )
        
        if success:
            print(f"Successfully added {len(embeddings)} vectors to the database")
        else:
            print("Failed to add all vectors to the database")
    
    if args.benchmark:
        # Run benchmark
        print("Running search benchmark...")
        benchmark_search(
            embeddings, coordinator_url, 
            num_queries=args.num_queries, 
            k_values=args.k_values
        )
    
    if args.search:
        # Perform a single search
        query_idx = np.random.randint(0, len(embeddings))
        query_vector = embeddings[query_idx]
        query_id = image_ids[query_idx]
        
        print(f"Searching for images similar to image ID: {query_id}")
        distances, indices = search_similar_images(
            query_vector, args.k, coordinator_url
        )
        
        if distances is not None and indices is not None:
            print(f"Found {len(indices)} similar images:")
            
            if len(indices) > 0:
                for i, (dist, idx) in enumerate(zip(distances, indices)):
                    # Format the distance value
                    if isinstance(dist, (int, float)):
                        dist_str = f"{dist:.4f}"
                    else:
                        dist_str = str(dist)
                        
                    print(f"  {i+1}. Image ID: {idx}, Distance: {dist_str}")
            else:
                print("  No similar images found.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load COCO embeddings and interact with the vector database")
    parser.add_argument("--embeddings", type=str, default="data/coco_embeddings.npz", help="Path to embeddings file")
    parser.add_argument("--host", type=str, default="localhost", help="Coordinator host")
    parser.add_argument("--port", type=int, default=6000, help="Coordinator port")
    parser.add_argument("--add", action="store_true", help="Add vectors to the database")
    parser.add_argument("--search", action="store_true", help="Perform a search")
    parser.add_argument("--benchmark", action="store_true", help="Run search benchmark")
    parser.add_argument("--batch-size", type=int, default=1000, help="Batch size for adding vectors")
    parser.add_argument("--k", type=int, default=10, help="Number of results to return for search")
    parser.add_argument("--num-queries", type=int, default=10, help="Number of queries for benchmark")
    parser.add_argument("--k-values", type=int, nargs="+", default=[1, 5, 10, 50, 100], help="k values for benchmark")
    parser.add_argument("--target-dim", type=int, default=512, help="Target dimension for embeddings")
    parser.add_argument("--reduction-method", type=str, default="pca", choices=["pca"], help="Dimension reduction method")
    
    args = parser.parse_args()
    main(args)
