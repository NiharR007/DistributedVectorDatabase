#!/usr/bin/env python
"""
Generate embeddings for COCO images using a pre-trained model.
"""

import os
import json
import numpy as np
import torch
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm
import argparse

def load_image(image_path):
    """Load and preprocess an image for the model."""
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)

def main(args):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device}")
    
    # Load pre-trained model
    if args.model == 'resnet50':
        model = models.resnet50(pretrained=True)
        # Remove the final classification layer
        model = torch.nn.Sequential(*list(model.children())[:-1])
    elif args.model == 'resnet101':
        model = models.resnet101(pretrained=True)
        model = torch.nn.Sequential(*list(model.children())[:-1])
    else:
        raise ValueError(f"Unsupported model: {args.model}")
    
    model.to(device)
    model.eval()
    
    # Get image paths
    image_dir = os.path.join(args.coco_dir, args.image_set)
    if not os.path.exists(image_dir):
        raise ValueError(f"Image directory not found: {image_dir}")
    
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
    print(f"Found {len(image_files)} images in {image_dir}")
    
    if args.limit > 0:
        image_files = image_files[:args.limit]
        print(f"Limited to {len(image_files)} images")
    
    # Generate embeddings
    embeddings = []
    image_ids = []
    
    with torch.no_grad():
        for image_file in tqdm(image_files, desc="Generating embeddings"):
            image_id = int(image_file.split('.')[0])
            image_path = os.path.join(image_dir, image_file)
            
            try:
                img_tensor = load_image(image_path)
                img_tensor = img_tensor.to(device)
                
                # Get embedding
                embedding = model(img_tensor)
                embedding = embedding.squeeze().cpu().numpy()
                
                embeddings.append(embedding)
                image_ids.append(image_id)
            except Exception as e:
                print(f"Error processing {image_file}: {e}")
    
    embeddings = np.array(embeddings)
    image_ids = np.array(image_ids)
    
    print(f"Generated embeddings with shape: {embeddings.shape}")
    
    # Save embeddings
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    np.savez(
        args.output,
        embeddings=embeddings,
        image_ids=image_ids
    )
    
    # Also save in JSON format for the vector database
    vectors_json = {
        "vectors": embeddings.tolist(),
        "ids": image_ids.tolist()
    }
    
    json_output = args.output.replace('.npz', '.json')
    with open(json_output, 'w') as f:
        json.dump(vectors_json, f)
    
    print(f"Saved embeddings to {args.output} and {json_output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate embeddings for COCO images")
    parser.add_argument("--coco-dir", type=str, default="data/coco", help="Path to COCO dataset directory")
    parser.add_argument("--image-set", type=str, default="val2017", help="Image set (train2017, val2017)")
    parser.add_argument("--output", type=str, default="data/coco_embeddings.npz", help="Output file path")
    parser.add_argument("--model", type=str, default="resnet50", choices=["resnet50", "resnet101"], help="Model to use")
    parser.add_argument("--limit", type=int, default=5000, help="Limit number of images (0 for no limit)")
    parser.add_argument("--cpu", action="store_true", help="Force CPU usage even if GPU is available")
    
    args = parser.parse_args()
    main(args)
