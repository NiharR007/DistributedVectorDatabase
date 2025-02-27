# Working with COCO Dataset in the Distributed Vector Database

This guide explains how to use the COCO (Common Objects in Context) dataset with your distributed vector database for a presentable big data problem.

## 1. Download COCO Dataset

First, download the COCO dataset and annotations:

```bash
# Create a directory for COCO data
mkdir -p data/coco

# Download COCO annotations
curl -L "http://images.cocodataset.org/annotations/annotations_trainval2017.zip" -o data/coco/annotations.zip
unzip data/coco/annotations.zip -d data/coco/

# Download a subset of COCO images (validation set)
curl -L "http://images.cocodataset.org/zips/val2017.zip" -o data/coco/val2017.zip
unzip data/coco/val2017.zip -d data/coco/
```

## 2. Generate Image Embeddings

Use the provided script to generate embeddings for the COCO images:

```bash
# Install additional dependencies if needed
pip install torch torchvision pillow tqdm

# Generate embeddings for 5000 images using ResNet50
python generate_coco_embeddings.py --coco-dir data/coco --image-set val2017 --limit 5000 --output data/coco_embeddings.npz
```

This will create two files:
- `data/coco_embeddings.npz`: NumPy format for fast loading
- `data/coco_embeddings.json`: JSON format for the vector database

## 3. Start the Distributed Vector Database

Start the database system as usual:

```bash
# Terminal 1: Start Shard Node 1
python api/shard_api.py 6001

# Terminal 2: Start Shard Node 2
python api/shard_api.py 6002

# Terminal 3: Start Coordinator
python api/coordinator_api.py
```

## 4. Load COCO Embeddings into the Database

Use the provided script to load the embeddings into your database:

```bash
# Add vectors to the database
python load_coco_embeddings.py --embeddings data/coco_embeddings.npz --add --batch-size 1000
```

## 5. Run Search Queries and Visualize Results

Search for similar images and visualize the results:

```bash
# Run multiple random queries and save visualizations
python visualize_results.py --embeddings data/coco_embeddings.npz --coco-dir data/coco --num-queries 5
```

This will create visualization images in the `visualizations` directory.

## 6. Analyze Performance

Analyze the performance of your distributed vector database:

```bash
# Analyze vector distribution across shards
python performance_analysis.py --embeddings data/coco_embeddings.npz --analyze-distribution

# Benchmark search latency vs k
python performance_analysis.py --embeddings data/coco_embeddings.npz --benchmark-k --num-queries 20
```

## 7. Generate a Presentation-Ready Report

Generate a PDF report with all the analysis and visualizations:

```bash
# Install additional dependency
pip install fpdf

# Generate report
python generate_report.py
```

This will create a file named `distributed_vector_db_report.pdf` with a comprehensive analysis of your system.

## 8. Presentation Ideas

When presenting this project, consider highlighting:

1. **System Architecture**: Explain the distributed nature of the database with coordinator and shard nodes.

2. **Sharding Strategy**: Discuss how LSH (Locality-Sensitive Hashing) is used to distribute vectors across shards.

3. **Similarity Search**: Demonstrate how the system finds similar images efficiently.

4. **Performance Analysis**: Show how the system scales with different parameters (k, number of vectors).

5. **Real-World Application**: Explain how this system could be used in production for image similarity search, content recommendation, or duplicate detection.

6. **CAP Theorem Tradeoffs**: Discuss how the system balances consistency, availability, and partition tolerance.

## 9. Troubleshooting

- **Memory Issues**: If you encounter memory problems with large datasets, reduce the number of vectors or use batching.
- **Slow Performance**: Adjust the LSH parameters in `config/coordinator.yaml` or index parameters in `config/shard_node.yaml`.
- **Connection Issues**: Ensure all services are running and ports are not blocked by firewalls.

## 10. Further Improvements

- Implement dynamic resharding to better balance vector distribution
- Add support for filtering based on metadata (e.g., image categories)
- Implement more advanced replication strategies for fault tolerance
- Explore hybrid indexing techniques for better recall/latency tradeoffs
