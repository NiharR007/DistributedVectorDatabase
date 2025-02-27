#!/usr/bin/env python
"""
Analyze the performance of the distributed vector database with COCO embeddings.
"""

import os
import json
import numpy as np
import requests
import argparse
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import seaborn as sns
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

def get_shard_stats(host, port):
    """Get statistics from a shard node."""
    url = f"http://{host}:{port}/stats"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error getting stats from {host}:{port}: {response.text}")
            return None
    except Exception as e:
        print(f"Exception getting stats from {host}:{port}: {e}")
        return None

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
        return None, None, None
    
    result = response.json()
    return result["distances"], result["indices"], response.elapsed.total_seconds() * 1000  # ms

def benchmark_latency_vs_k(embeddings, coordinator_url, num_queries=10, k_values=None):
    """Benchmark search latency vs k."""
    if k_values is None:
        k_values = [1, 5, 10, 20, 50, 100]
    
    # Select random query vectors
    query_indices = np.random.choice(len(embeddings), num_queries, replace=False)
    query_vectors = embeddings[query_indices]
    
    results = {k: [] for k in k_values}
    
    for k in k_values:
        print(f"Testing k={k}")
        for i, query_vector in enumerate(tqdm(query_vectors, desc=f"Queries for k={k}")):
            _, _, latency = search_similar_images(query_vector, k, coordinator_url)
            if latency is not None:
                results[k].append(latency)
    
    # Calculate statistics
    stats = {
        'k': k_values,
        'mean_latency': [np.mean(results[k]) for k in k_values],
        'median_latency': [np.median(results[k]) for k in k_values],
        'min_latency': [np.min(results[k]) for k in k_values],
        'max_latency': [np.max(results[k]) for k in k_values],
        'std_latency': [np.std(results[k]) for k in k_values]
    }
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.errorbar(
        stats['k'], stats['mean_latency'], 
        yerr=stats['std_latency'], 
        fmt='o-', capsize=5
    )
    plt.xlabel('k (number of results)')
    plt.ylabel('Latency (ms)')
    plt.title('Search Latency vs. k')
    plt.grid(True)
    plt.savefig('latency_vs_k.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create a more detailed plot
    df = pd.DataFrame({
        'k': np.repeat(k_values, [len(results[k]) for k in k_values]),
        'latency': np.concatenate([results[k] for k in k_values])
    })
    
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='k', y='latency', data=df)
    plt.xlabel('k (number of results)')
    plt.ylabel('Latency (ms)')
    plt.title('Search Latency Distribution vs. k')
    plt.grid(True, axis='y')
    plt.savefig('latency_distribution_vs_k.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return stats

def benchmark_latency_vs_vectors(embeddings, coordinator_url, num_queries=10, k=10, vector_counts=None):
    """Benchmark search latency vs number of vectors in the database."""
    if vector_counts is None:
        vector_counts = [1000, 5000, 10000, 20000, 50000]
    
    # Ensure we have enough vectors
    if len(embeddings) < max(vector_counts):
        print(f"Warning: Not enough vectors for the largest count ({max(vector_counts)})")
        vector_counts = [vc for vc in vector_counts if vc <= len(embeddings)]
    
    # Select random query vectors (not in the database)
    query_indices = np.random.choice(len(embeddings), num_queries, replace=False)
    query_vectors = embeddings[query_indices]
    
    results = {vc: [] for vc in vector_counts}
    
    # For each vector count
    for vc in vector_counts:
        print(f"Testing with {vc} vectors in the database")
        
        # Add vectors to the database
        # (This assumes you have a way to reset the database between tests)
        
        # Run queries
        for i, query_vector in enumerate(tqdm(query_vectors, desc=f"Queries for {vc} vectors")):
            _, _, latency = search_similar_images(query_vector, k, coordinator_url)
            if latency is not None:
                results[vc].append(latency)
    
    # Calculate statistics
    stats = {
        'vector_count': vector_counts,
        'mean_latency': [np.mean(results[vc]) for vc in vector_counts],
        'median_latency': [np.median(results[vc]) for vc in vector_counts],
        'min_latency': [np.min(results[vc]) for vc in vector_counts],
        'max_latency': [np.max(results[vc]) for vc in vector_counts],
        'std_latency': [np.std(results[vc]) for vc in vector_counts]
    }
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.errorbar(
        stats['vector_count'], stats['mean_latency'], 
        yerr=stats['std_latency'], 
        fmt='o-', capsize=5
    )
    plt.xlabel('Number of Vectors')
    plt.ylabel('Latency (ms)')
    plt.title(f'Search Latency vs. Database Size (k={k})')
    plt.grid(True)
    plt.savefig('latency_vs_vectors.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return stats

def analyze_shard_distribution(coordinator_config, shard_ports):
    """Analyze the distribution of vectors across shards."""
    # Get stats from each shard
    shard_stats = []
    for port in shard_ports:
        stats = get_shard_stats('localhost', port)
        if stats:
            stats['port'] = port
            shard_stats.append(stats)
    
    if not shard_stats:
        print("No shard statistics available")
        return
    
    # Extract vector counts
    ports = [stats['port'] for stats in shard_stats]
    vector_counts = [stats['total_vectors'] for stats in shard_stats]
    
    # Plot distribution
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(ports)), vector_counts)
    plt.xlabel('Shard')
    plt.ylabel('Number of Vectors')
    plt.title('Vector Distribution Across Shards')
    plt.xticks(range(len(ports)), [f"Shard {i+1}\n(Port {port})" for i, port in enumerate(ports)])
    plt.grid(True, axis='y')
    
    # Add text labels on top of bars
    for i, count in enumerate(vector_counts):
        plt.text(i, count + max(vector_counts) * 0.01, str(count), 
                 ha='center', va='bottom', fontweight='bold')
    
    plt.savefig('shard_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Calculate distribution metrics
    total_vectors = sum(vector_counts)
    distribution_percentages = [count / total_vectors * 100 for count in vector_counts]
    
    print("Vector Distribution Across Shards:")
    for i, (port, count, percentage) in enumerate(zip(ports, vector_counts, distribution_percentages)):
        print(f"  Shard {i+1} (Port {port}): {count} vectors ({percentage:.2f}%)")
    
    # Calculate imbalance metrics
    if len(vector_counts) > 1:
        min_count = min(vector_counts)
        max_count = max(vector_counts)
        mean_count = np.mean(vector_counts)
        std_count = np.std(vector_counts)
        cv = std_count / mean_count if mean_count > 0 else 0
        
        print(f"\nDistribution Metrics:")
        print(f"  Min vectors: {min_count}")
        print(f"  Max vectors: {max_count}")
        print(f"  Mean vectors: {mean_count:.2f}")
        print(f"  Standard deviation: {std_count:.2f}")
        print(f"  Coefficient of variation: {cv:.4f}")
        print(f"  Max/Min ratio: {max_count / min_count if min_count > 0 else 'N/A'}")

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
    
    if args.analyze_distribution:
        # Analyze shard distribution
        print("Analyzing shard distribution...")
        analyze_shard_distribution(
            args.coordinator_config, 
            args.shard_ports
        )
    
    if args.benchmark_k:
        # Benchmark latency vs k
        print("Benchmarking latency vs k...")
        benchmark_latency_vs_k(
            embeddings, coordinator_url,
            args.num_queries, args.k_values
        )
    
    if args.benchmark_vectors:
        # Benchmark latency vs vector count
        print("Benchmarking latency vs vector count...")
        benchmark_latency_vs_vectors(
            embeddings, coordinator_url,
            args.num_queries, args.k,
            args.vector_counts
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze performance of the distributed vector database")
    parser.add_argument("--embeddings", type=str, default="data/coco_embeddings.npz", help="Path to embeddings file")
    parser.add_argument("--host", type=str, default="localhost", help="Coordinator host")
    parser.add_argument("--port", type=int, default=6000, help="Coordinator port")
    parser.add_argument("--coordinator-config", type=str, default="config/coordinator.yaml", help="Path to coordinator config")
    parser.add_argument("--shard-ports", type=int, nargs="+", default=[6001, 6002], help="Ports of shard nodes")
    parser.add_argument("--analyze-distribution", action="store_true", help="Analyze vector distribution across shards")
    parser.add_argument("--benchmark-k", action="store_true", help="Benchmark latency vs k")
    parser.add_argument("--benchmark-vectors", action="store_true", help="Benchmark latency vs vector count")
    parser.add_argument("--num-queries", type=int, default=10, help="Number of queries for benchmarks")
    parser.add_argument("--k", type=int, default=10, help="k for vector count benchmark")
    parser.add_argument("--k-values", type=int, nargs="+", default=[1, 5, 10, 20, 50, 100], help="k values for benchmark")
    parser.add_argument("--vector-counts", type=int, nargs="+", default=[1000, 5000, 10000, 20000, 50000], help="Vector counts for benchmark")
    parser.add_argument("--target-dim", type=int, default=512, help="Target dimension for embeddings")
    parser.add_argument("--reduction-method", type=str, default="pca", choices=["pca"], help="Dimension reduction method")
    
    args = parser.parse_args()
    main(args)
