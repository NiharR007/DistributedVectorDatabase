#!/usr/bin/env python
"""
Search for similar images by providing an input image.
"""

import os
import sys
import json
import numpy as np
import requests
import argparse
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm
from sklearn.decomposition import PCA
import torch
from torchvision import models, transforms
from visualize_results import get_image_path, visualize_search_results

# Set up image preprocessing
def get_image_transform():
    """Get image transformation pipeline for feature extraction."""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def load_feature_extractor():
    """Load pre-trained ResNet model for feature extraction."""
    # Load pre-trained ResNet model
    model = models.resnet50(pretrained=True)
    
    # Remove the final classification layer
    feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])
    feature_extractor.eval()
    
    return feature_extractor

def extract_features(image_path, feature_extractor, transform):
    """Extract features from an image using the pre-trained model."""
    # Load and preprocess the image
    try:
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0)
        
        # Extract features
        with torch.no_grad():
            features = feature_extractor(image_tensor)
            
        # Reshape and convert to numpy array
        features = features.squeeze().flatten().numpy()
        return features
    except Exception as e:
        print(f"Error extracting features from {image_path}: {str(e)}")
        return None

def load_pca_model(embeddings_file, target_dim=512):
    """Load embeddings and train a PCA model."""
    # Load embeddings
    data = np.load(embeddings_file)
    embeddings = data['embeddings']
    
    # Train PCA model
    pca = PCA(n_components=target_dim)
    pca.fit(embeddings)
    
    explained_variance = sum(pca.explained_variance_ratio_) * 100
    print(f"PCA explained variance: {explained_variance:.2f}%")
    
    return pca

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
    except Exception as e:
        print(f"Error during search: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return None, None

def visualize_input_and_results(input_image_path, similar_ids, distances, coco_dir, image_set='val2017', output_file=None):
    """Visualize input image and similar images."""
    n_results = len(similar_ids)
    
    # Handle case when no results are returned
    if n_results == 0:
        print(f"Warning: No similar images found for input image: {input_image_path}")
        # Create figure with just the input image
        fig = plt.figure(figsize=(8, 8))
        
        # Plot input image
        if os.path.exists(input_image_path):
            input_img = Image.open(input_image_path)
            plt.imshow(input_img)
            plt.title(f"Input Image\nNo similar images found")
            plt.axis('off')
        else:
            print(f"Input image not found: {input_image_path}")
            
        if output_file:
            plt.savefig(output_file, bbox_inches='tight')
            print(f"Saved visualization to {output_file}")
        else:
            plt.show()
        
        plt.close()
        return
    
    # Create figure with grid
    fig = plt.figure(figsize=(15, 8))
    gs = gridspec.GridSpec(2, (n_results + 1) // 2)
    
    # Plot input image
    if os.path.exists(input_image_path):
        input_img = Image.open(input_image_path)
        ax = plt.subplot(gs[0, 0])
        ax.imshow(input_img)
        ax.set_title(f"Input Image")
        ax.axis('off')
    else:
        print(f"Input image not found: {input_image_path}")
    
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

def main(args):
    # Load feature extractor
    print("Loading feature extractor...")
    feature_extractor = load_feature_extractor()
    transform = get_image_transform()
    
    # Extract features from input image
    print(f"Extracting features from input image: {args.input_image}")
    features = extract_features(args.input_image, feature_extractor, transform)
    
    if features is None:
        print("Failed to extract features from input image")
        return
    
    print(f"Extracted features with shape: {features.shape}")
    
    # Load PCA model and reduce dimensions
    print(f"Loading PCA model from embeddings: {args.embeddings}")
    pca = load_pca_model(args.embeddings, args.target_dim)
    
    # Reduce dimensions
    reduced_features = pca.transform(features.reshape(1, -1))[0]
    print(f"Reduced features shape: {reduced_features.shape}")
    
    # Search for similar images
    coordinator_url = f"http://{args.host}:{args.port}"
    print(f"Searching for similar images...")
    distances, similar_ids = search_similar_images(
        reduced_features, args.k, coordinator_url
    )
    
    if distances is not None and similar_ids is not None and len(distances) > 0:
        print(f"Found {len(similar_ids)} similar images")
        
        # Visualize results
        output_file = args.output if args.output else "similar_images.png"
        visualize_input_and_results(
            args.input_image, similar_ids, distances, 
            args.coco_dir, args.image_set, output_file
        )
    else:
        print("No similar images found")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Search for similar images by providing an input image")
    parser.add_argument("--input-image", type=str, required=True, help="Path to input image")
    parser.add_argument("--embeddings", type=str, default="data/coco_embeddings.npz", help="Path to embeddings file")
    parser.add_argument("--coco-dir", type=str, default="data/coco", help="Path to COCO dataset directory")
    parser.add_argument("--image-set", type=str, default="val2017", help="Image set (train2017, val2017)")
    parser.add_argument("--host", type=str, default="localhost", help="Coordinator host")
    parser.add_argument("--port", type=int, default=6000, help="Coordinator port")
    parser.add_argument("--k", type=int, default=5, help="Number of similar images to find")
    parser.add_argument("--output", type=str, help="Output file for visualization")
    parser.add_argument("--target-dim", type=int, default=512, help="Target dimension for embeddings")
    
    args = parser.parse_args()
    main(args)
