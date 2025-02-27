#!/usr/bin/env python
"""
Run the complete distributed vector database system with COCO dataset.
This script orchestrates the entire workflow:
1. Generate embeddings from COCO images
2. Load embeddings into the distributed database
3. Start the coordinator and shard nodes
4. Run search queries and visualize results
5. Analyze performance
6. Generate a comprehensive report
"""

import os
import argparse
import subprocess
import time
import yaml
import shutil
from pathlib import Path

def ensure_directory(directory):
    """Ensure directory exists."""
    Path(directory).mkdir(parents=True, exist_ok=True)

def run_command(command, description=None):
    """Run a command and print its output."""
    if description:
        print(f"\n{description}...")
    
    print(f"Running: {' '.join(command)}")
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True
    )
    
    for line in process.stdout:
        print(line, end='')
    
    process.wait()
    return process.returncode

def start_server(command, log_file, description=None):
    """Start a server process and return its process object."""
    if description:
        print(f"\n{description}...")
    
    print(f"Starting: {' '.join(command)}")
    log_fd = open(log_file, 'w')
    process = subprocess.Popen(
        command,
        stdout=log_fd,
        stderr=log_fd
    )
    
    # Give it a moment to start
    time.sleep(2)
    
    if process.poll() is not None:
        print(f"Failed to start process. Check log file: {log_file}")
        return None
    
    print(f"Started process with PID {process.pid}")
    return process

def main(args):
    # Create necessary directories
    ensure_directory("data")
    ensure_directory("logs")
    ensure_directory("visualizations")
    ensure_directory("config")
    
    # Generate default config files if they don't exist
    if not os.path.exists(args.coordinator_config):
        coordinator_config = {
            "host": "localhost",
            "port": args.coordinator_port,
            "sharding_strategy": "lsh",
            "replication_factor": 1,
            "consistency_mode": "eventual",
            "shard_nodes": [
                {"host": "localhost", "port": port} for port in args.shard_ports
            ]
        }
        
        with open(args.coordinator_config, 'w') as f:
            yaml.dump(coordinator_config, f)
    
    if not os.path.exists(args.shard_config):
        shard_config = {
            "index_type": "hnsw",
            "distance_metric": "cosine",
            "performance_tracking": True,
            "vector_dimension": args.target_dim
        }
        
        with open(args.shard_config, 'w') as f:
            yaml.dump(shard_config, f)
    
    # Step 1: Generate embeddings if needed
    if not os.path.exists(args.embeddings) or args.force_generate:
        run_command([
            "python", "generate_coco_embeddings.py",
            "--coco-dir", args.coco_dir,
            "--output", args.embeddings,
            "--model", args.model,
            "--batch-size", str(args.batch_size),
            "--device", args.device
        ], "Generating embeddings from COCO images")
    
    # Step 2: Start shard nodes
    shard_processes = []
    for i, port in enumerate(args.shard_ports):
        process = start_server([
            "python", "shard/shard_server.py",
            "--host", "localhost",
            "--port", str(port),
            "--config", args.shard_config,
            "--data-dir", f"data/shard_{i}"
        ], f"logs/shard_{port}.log", f"Starting shard node {i+1} on port {port}")
        
        if process:
            shard_processes.append(process)
    
    if not shard_processes:
        print("Failed to start any shard nodes. Exiting.")
        return 1
    
    # Step 3: Start coordinator
    coordinator_process = start_server([
        "python", "coordinator/coordinator_server.py",
        "--config", args.coordinator_config
    ], "logs/coordinator.log", f"Starting coordinator on port {args.coordinator_port}")
    
    if not coordinator_process:
        print("Failed to start coordinator. Stopping shard nodes and exiting.")
        for process in shard_processes:
            process.terminate()
        return 1
    
    # Give servers a moment to initialize
    print("Waiting for servers to initialize...")
    time.sleep(5)
    
    # Step 4: Load embeddings into database
    if args.load_embeddings:
        run_command([
            "python", "load_coco_embeddings.py",
            "--embeddings", args.embeddings,
            "--host", "localhost",
            "--port", str(args.coordinator_port),
            "--batch-size", str(args.batch_size),
            "--target-dim", str(args.target_dim),
            "--reduction-method", args.reduction_method
        ], "Loading embeddings into database")
    
    # Step 5: Run search and visualize results
    if args.run_search:
        run_command([
            "python", "visualize_results.py",
            "--embeddings", args.embeddings,
            "--coco-dir", args.coco_dir,
            "--host", "localhost",
            "--port", str(args.coordinator_port),
            "--num-queries", str(args.num_queries),
            "--k", str(args.k),
            "--output-dir", "visualizations",
            "--target-dim", str(args.target_dim),
            "--reduction-method", args.reduction_method
        ], "Running search queries and visualizing results")
    
    # Step 6: Analyze performance
    if args.analyze_performance:
        run_command([
            "python", "performance_analysis.py",
            "--embeddings", args.embeddings,
            "--host", "localhost",
            "--port", str(args.coordinator_port),
            "--shard-ports", *[str(port) for port in args.shard_ports],
            "--analyze-distribution",
            "--benchmark-k",
            "--num-queries", str(args.num_queries),
            "--k", str(args.k),
            "--target-dim", str(args.target_dim),
            "--reduction-method", args.reduction_method
        ], "Analyzing performance")
    
    # Step 7: Generate report
    if args.generate_report:
        run_command([
            "python", "generate_report.py",
            "--coordinator-config", args.coordinator_config,
            "--shard-config", args.shard_config,
            "--embeddings", args.embeddings,
            "--visualization-dir", "visualizations",
            "--target-dim", str(args.target_dim),
            "--reduction-method", args.reduction_method,
            "--output", args.report_output
        ], "Generating comprehensive report")
    
    # Keep servers running if requested
    if args.keep_running:
        print("\nServers are running. Press Ctrl+C to stop...")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("Stopping servers...")
    
    # Stop servers
    if coordinator_process and coordinator_process.poll() is None:
        print("Stopping coordinator...")
        coordinator_process.terminate()
    
    for i, process in enumerate(shard_processes):
        if process.poll() is None:
            print(f"Stopping shard node {i+1}...")
            process.terminate()
    
    print("All done!")
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the complete distributed vector database system")
    
    # General options
    parser.add_argument("--coco-dir", type=str, default="data/coco", help="Path to COCO dataset directory")
    parser.add_argument("--embeddings", type=str, default="data/coco_embeddings.npz", help="Path to embeddings file")
    parser.add_argument("--coordinator-config", type=str, default="config/coordinator.yaml", help="Path to coordinator config")
    parser.add_argument("--shard-config", type=str, default="config/shard_node.yaml", help="Path to shard node config")
    parser.add_argument("--coordinator-port", type=int, default=6000, help="Coordinator port")
    parser.add_argument("--shard-ports", type=int, nargs="+", default=[6001, 6002, 6003], help="Shard node ports")
    parser.add_argument("--keep-running", action="store_true", help="Keep servers running after completion")
    
    # Embedding generation options
    parser.add_argument("--force-generate", action="store_true", help="Force regenerate embeddings even if they exist")
    parser.add_argument("--model", type=str, default="resnet50", choices=["resnet50", "resnet101"], help="Model for generating embeddings")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Device for generating embeddings")
    
    # Dimension reduction options
    parser.add_argument("--target-dim", type=int, default=512, help="Target dimension for embeddings")
    parser.add_argument("--reduction-method", type=str, default="pca", choices=["pca"], help="Dimension reduction method")
    
    # Workflow control
    parser.add_argument("--load-embeddings", action="store_true", help="Load embeddings into database")
    parser.add_argument("--run-search", action="store_true", help="Run search queries and visualize results")
    parser.add_argument("--analyze-performance", action="store_true", help="Analyze performance")
    parser.add_argument("--generate-report", action="store_true", help="Generate comprehensive report")
    parser.add_argument("--run-all", action="store_true", help="Run all steps")
    
    # Performance parameters
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for processing")
    parser.add_argument("--num-queries", type=int, default=5, help="Number of random queries to run")
    parser.add_argument("--k", type=int, default=5, help="Number of similar images to find")
    
    # Output
    parser.add_argument("--report-output", type=str, default="distributed_vector_db_report.pdf", help="Output file for report")
    
    args = parser.parse_args()
    
    # If run-all is specified, enable all workflow steps
    if args.run_all:
        args.load_embeddings = True
        args.run_search = True
        args.analyze_performance = True
        args.generate_report = True
    
    main(args)
