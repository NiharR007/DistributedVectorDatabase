#!/usr/bin/env python
"""
Generate a presentation-ready report for the distributed vector database.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import argparse
from datetime import datetime
import requests
from PIL import Image
from fpdf import FPDF
import yaml

class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Distributed Vector Database Performance Report', 0, 1, 'C')
        self.ln(5)
    
    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

def get_system_info(coordinator_url, shard_ports):
    """Get information about the system configuration."""
    system_info = {
        'coordinator': None,
        'shards': []
    }
    
    # Try to get coordinator stats
    try:
        response = requests.get(f"{coordinator_url}/stats")
        if response.status_code == 200:
            system_info['coordinator'] = response.json()
    except Exception as e:
        print(f"Error getting coordinator stats: {e}")
    
    # Get shard stats
    for port in shard_ports:
        try:
            response = requests.get(f"http://localhost:{port}/stats")
            if response.status_code == 200:
                shard_info = response.json()
                shard_info['port'] = port
                system_info['shards'].append(shard_info)
        except Exception as e:
            print(f"Error getting stats from shard on port {port}: {e}")
    
    return system_info

def load_config(config_file):
    """Load configuration from YAML file."""
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)

def load_embeddings_info(embeddings_file):
    """Load information about embeddings."""
    if not os.path.exists(embeddings_file):
        return None
        
    try:
        data = np.load(embeddings_file)
        info = {
            'num_embeddings': len(data['embeddings']),
            'embedding_dim': data['embeddings'].shape[1],
            'image_ids': data['image_ids']
        }
        return info
    except Exception as e:
        print(f"Error loading embeddings info: {e}")
        return None

def generate_system_overview(pdf, system_info, coordinator_config, shard_config, embeddings_info, target_dim, reduction_method):
    """Generate system overview section."""
    pdf.add_page()
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, '1. System Overview', 0, 1)
    pdf.ln(5)
    
    # Coordinator configuration
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, '1.1 Coordinator Configuration', 0, 1)
    pdf.set_font('Arial', '', 10)
    
    if coordinator_config:
        pdf.multi_cell(0, 10, f"Sharding Strategy: {coordinator_config.get('sharding_strategy', 'N/A')}")
        pdf.multi_cell(0, 10, f"Replication Factor: {coordinator_config.get('replication_factor', 'N/A')}")
        pdf.multi_cell(0, 10, f"Consistency Mode: {coordinator_config.get('consistency_mode', 'N/A')}")
        
        # LSH configuration if available
        lsh_config = coordinator_config.get('lsh_config', {})
        if lsh_config:
            pdf.multi_cell(0, 10, f"LSH Hash Functions: {lsh_config.get('num_hash_functions', 'N/A')}")
            pdf.multi_cell(0, 10, f"LSH Hash Tables: {lsh_config.get('num_hash_tables', 'N/A')}")
    else:
        pdf.multi_cell(0, 10, "Coordinator configuration not available")
    
    # Shard configuration
    pdf.ln(5)
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, '1.2 Shard Configuration', 0, 1)
    pdf.set_font('Arial', '', 10)
    
    if shard_config:
        pdf.multi_cell(0, 10, f"Index Type: {shard_config.get('index_type', 'N/A')}")
        
        index_params = shard_config.get('index_params', {})
        if index_params:
            for param, value in index_params.items():
                pdf.multi_cell(0, 10, f"{param}: {value}")
    else:
        pdf.multi_cell(0, 10, "Shard configuration not available")
    
    # System statistics
    pdf.ln(5)
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, '1.3 Current System State', 0, 1)
    pdf.set_font('Arial', '', 10)
    
    if system_info['shards']:
        total_vectors = sum(shard.get('total_vectors', 0) for shard in system_info['shards'])
        pdf.multi_cell(0, 10, f"Total Vectors: {total_vectors}")
        pdf.multi_cell(0, 10, f"Number of Shards: {len(system_info['shards'])}")
        
        # Vector dimension
        if system_info['shards'][0].get('dimension'):
            pdf.multi_cell(0, 10, f"Vector Dimension: {system_info['shards'][0]['dimension']}")
        
        # Distribution table
        pdf.ln(5)
        pdf.set_font('Arial', 'B', 10)
        pdf.cell(30, 10, 'Shard', 1)
        pdf.cell(30, 10, 'Port', 1)
        pdf.cell(40, 10, 'Vectors', 1)
        pdf.cell(40, 10, 'Percentage', 1)
        pdf.ln()
        
        pdf.set_font('Arial', '', 10)
        for i, shard in enumerate(system_info['shards']):
            vectors = shard.get('total_vectors', 0)
            percentage = (vectors / total_vectors * 100) if total_vectors > 0 else 0
            
            pdf.cell(30, 10, f"Shard {i+1}", 1)
            pdf.cell(30, 10, f"{shard.get('port', 'N/A')}", 1)
            pdf.cell(40, 10, f"{vectors}", 1)
            pdf.cell(40, 10, f"{percentage:.2f}%", 1)
            pdf.ln()
    else:
        pdf.multi_cell(0, 10, "No shard statistics available")
    
    # Dimension reduction
    pdf.ln(5)
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, '1.4 Dimension Reduction', 0, 1)
    pdf.set_font('Arial', '', 10)
    
    if embeddings_info:
        pdf.multi_cell(0, 10, f"Original Embedding Dimension: {embeddings_info['embedding_dim']}")
        pdf.multi_cell(0, 10, f"Target Dimension after Reduction: {target_dim}")
        pdf.multi_cell(0, 10, f"Dimension Reduction Method: {reduction_method.upper()}")
        
        pdf.multi_cell(0, 10, f"\nDimension reduction from {embeddings_info['embedding_dim']} to {target_dim} dimensions using {reduction_method.upper()} maintains the semantic relationships between images while significantly improving storage efficiency and search performance.")
    else:
        pdf.multi_cell(0, 10, "Dimension reduction information not available")

def generate_performance_section(pdf, performance_data):
    """Generate performance analysis section."""
    pdf.add_page()
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, '2. Performance Analysis', 0, 1)
    pdf.ln(5)
    
    # Latency vs k
    if os.path.exists('latency_vs_k.png'):
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, '2.1 Search Latency vs. k', 0, 1)
        pdf.image('latency_vs_k.png', x=10, w=180)
        pdf.ln(5)
        pdf.set_font('Arial', '', 10)
        pdf.multi_cell(0, 10, "This graph shows how search latency scales with the number of results (k) requested. As k increases, the search operation needs to find and return more similar vectors, which can increase processing time.")
    
    # Latency distribution
    if os.path.exists('latency_distribution_vs_k.png'):
        pdf.ln(5)
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, '2.2 Latency Distribution', 0, 1)
        pdf.image('latency_distribution_vs_k.png', x=10, w=180)
        pdf.ln(5)
        pdf.set_font('Arial', '', 10)
        pdf.multi_cell(0, 10, "This box plot shows the distribution of search latencies for different k values. The box represents the interquartile range (IQR), the line inside the box is the median, and the whiskers extend to 1.5 times the IQR. Outliers are shown as individual points.")
    
    # Shard distribution
    if os.path.exists('shard_distribution.png'):
        pdf.ln(5)
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, '2.3 Vector Distribution Across Shards', 0, 1)
        pdf.image('shard_distribution.png', x=10, w=180)
        pdf.ln(5)
        pdf.set_font('Arial', '', 10)
        pdf.multi_cell(0, 10, "This chart shows how vectors are distributed across different shards. An ideal distribution would have vectors evenly spread across all shards to balance the workload and maximize parallel processing capabilities.")

def generate_search_examples(pdf, example_images_dir):
    """Generate search examples section with visualizations."""
    if not os.path.exists(example_images_dir):
        return
    
    example_images = [f for f in os.listdir(example_images_dir) if f.endswith('.png')]
    if not example_images:
        return
    
    pdf.add_page()
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, '3. Search Examples', 0, 1)
    pdf.ln(5)
    
    pdf.set_font('Arial', '', 10)
    pdf.multi_cell(0, 10, "This section shows examples of similarity search results. For each query image, we show the top similar images found by the distributed vector database system.")
    
    for i, image_file in enumerate(example_images[:3]):  # Limit to 3 examples
        pdf.ln(5)
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, f'3.{i+1} Example {i+1}', 0, 1)
        
        image_path = os.path.join(example_images_dir, image_file)
        pdf.image(image_path, x=10, w=180)
        
        pdf.ln(5)
        pdf.set_font('Arial', '', 10)
        pdf.multi_cell(0, 10, f"Query results for image {image_file.split('_')[1]}. The leftmost image is the query, and the other images are the most similar vectors found in the database.")

def generate_conclusion(pdf, system_info, embeddings_info, target_dim, reduction_method):
    """Generate conclusion section."""
    pdf.add_page()
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, '4. Conclusion', 0, 1)
    pdf.ln(5)
    
    pdf.set_font('Arial', '', 10)
    
    # Calculate some metrics
    if system_info['shards']:
        total_vectors = sum(shard.get('total_vectors', 0) for shard in system_info['shards'])
        num_shards = len(system_info['shards'])
        
        vector_counts = [shard.get('total_vectors', 0) for shard in system_info['shards']]
        min_count = min(vector_counts) if vector_counts else 0
        max_count = max(vector_counts) if vector_counts else 0
        mean_count = np.mean(vector_counts) if vector_counts else 0
        std_count = np.std(vector_counts) if vector_counts else 0
        cv = std_count / mean_count if mean_count > 0 else 0
        
        pdf.multi_cell(0, 10, f"Our distributed vector database successfully stores {total_vectors} high-dimensional image embeddings across {num_shards} shards.")
        
        if cv < 0.1:
            distribution_quality = "excellent"
        elif cv < 0.2:
            distribution_quality = "good"
        elif cv < 0.3:
            distribution_quality = "acceptable"
        else:
            distribution_quality = "uneven"
        
        pdf.multi_cell(0, 10, f"The vector distribution is {distribution_quality} with a coefficient of variation of {cv:.4f}. The most populated shard contains {max_count} vectors, while the least populated contains {min_count} vectors.")
    else:
        pdf.multi_cell(0, 10, "The distributed vector database is operational, but detailed statistics are not available.")
    
    pdf.ln(5)
    pdf.multi_cell(0, 10, "The system demonstrates the following key capabilities:")
    pdf.ln(2)
    
    capabilities = [
        "Horizontal sharding of vector data across multiple nodes",
        "Efficient similarity search using approximate nearest neighbor algorithms",
        "Distributed query execution with result merging",
        "Scalable architecture that can handle large datasets",
        "Support for high-dimensional image embeddings"
    ]
    
    for capability in capabilities:
        pdf.cell(10, 10, "•", 0)
        pdf.multi_cell(0, 10, capability)
    
    pdf.ln(5)
    pdf.multi_cell(0, 10, "Future improvements could include:")
    pdf.ln(2)
    
    improvements = [
        "Dynamic resharding to better balance the vector distribution",
        "Advanced replication strategies for improved fault tolerance",
        "Hybrid indexing techniques for better recall/latency tradeoffs",
        "Support for vector filtering and metadata queries",
        "Integration with real-time data streams"
    ]
    
    for improvement in improvements:
        pdf.cell(10, 10, "•", 0)
        pdf.multi_cell(0, 10, improvement)
    
    pdf.ln(5)
    pdf.multi_cell(0, 10, f"The dimension reduction from {embeddings_info['embedding_dim'] if embeddings_info else 'original dimension'} to {target_dim} dimensions using {reduction_method.upper()} maintains the semantic relationships between images while significantly improving storage efficiency and search performance.")

def main(args):
    # Get system information
    coordinator_url = f"http://{args.host}:{args.port}"
    system_info = get_system_info(coordinator_url, args.shard_ports)
    
    # Load configurations
    coordinator_config = None
    shard_config = None
    
    if os.path.exists(args.coordinator_config):
        coordinator_config = load_config(args.coordinator_config)
    
    if os.path.exists(args.shard_config):
        shard_config = load_config(args.shard_config)
    
    # Load embeddings info
    embeddings_info = load_embeddings_info(args.embeddings)
    
    # Create PDF
    pdf = PDF()
    pdf.set_title('Distributed Vector Database Performance Report')
    pdf.set_author('Advanced DBMS Project')
    
    # Generate sections
    generate_system_overview(pdf, system_info, coordinator_config, shard_config, embeddings_info, args.target_dim, args.reduction_method)
    generate_performance_section(pdf, None)  # We'll use the saved images
    generate_search_examples(pdf, args.visualizations_dir)
    generate_conclusion(pdf, system_info, embeddings_info, args.target_dim, args.reduction_method)
    
    # Save PDF
    pdf.output(args.output)
    print(f"Report generated: {args.output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a presentation-ready report")
    parser.add_argument("--host", type=str, default="localhost", help="Coordinator host")
    parser.add_argument("--port", type=int, default=6000, help="Coordinator port")
    parser.add_argument("--shard-ports", type=int, nargs="+", default=[6001, 6002], help="Ports of shard nodes")
    parser.add_argument("--coordinator-config", type=str, default="config/coordinator.yaml", help="Path to coordinator config")
    parser.add_argument("--shard-config", type=str, default="config/shard_node.yaml", help="Path to shard config")
    parser.add_argument("--embeddings", type=str, default="data/coco_embeddings.npz", help="Path to embeddings file")
    parser.add_argument("--visualizations-dir", type=str, default="visualizations", help="Directory with visualization images")
    parser.add_argument("--target-dim", type=int, default=512, help="Target dimension for embeddings")
    parser.add_argument("--reduction-method", type=str, default="pca", help="Dimension reduction method")
    parser.add_argument("--output", type=str, default="distributed_vector_db_report.pdf", help="Output PDF file")
    
    args = parser.parse_args()
    main(args)
