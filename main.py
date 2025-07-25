#!/usr/bin/env python3
"""
FastServe Main Entry Point
Start the FastServe server with command-line options
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from fastserve.server import run_server
from fastserve.config import get_config, update_config

def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration"""
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {log_level}')
    
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('fastserve.log')
        ]
    )

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="FastServe - Fast Distributed Inference Serving for Large Language Models"
    )
    
    # Server options
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")
    
    # Model options
    parser.add_argument("--model", help="Model name or path")
    parser.add_argument("--max-length", type=int, help="Maximum sequence length")
    parser.add_argument("--max-new-tokens", type=int, help="Maximum new tokens to generate")
    
    # Performance options
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], 
                       default="auto", help="Device to use")
    parser.add_argument("--batch-size", type=int, help="Batch size")
    parser.add_argument("--cache-size", type=int, help="Cache size in MB")
    
    # Scheduler options
    parser.add_argument("--num-queues", type=int, help="Number of priority queues")
    parser.add_argument("--queue-capacity", type=int, help="Capacity per queue")
    parser.add_argument("--disable-preemption", action="store_true", 
                       help="Disable preemptive scheduling")
    
    # Logging
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       default="INFO", help="Logging level")
    
    # Development options
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # Update configuration based on arguments
    config_updates = {}
    
    if args.model:
        config_updates["model_name"] = args.model
    if args.max_length:
        config_updates["max_sequence_length"] = args.max_length
    if args.max_new_tokens:
        config_updates["max_new_tokens"] = args.max_new_tokens
    if args.batch_size:
        config_updates["batch_size"] = args.batch_size
    if args.cache_size:
        config_updates["max_cache_size_mb"] = args.cache_size
    if args.num_queues:
        config_updates["num_priority_queues"] = args.num_queues
    if args.queue_capacity:
        config_updates["queue_capacity"] = args.queue_capacity
    if args.disable_preemption:
        config_updates["preemption_enabled"] = False
    
    # Handle device selection
    if args.device != "auto":
        config_updates["device"] = args.device
    
    # Apply configuration updates
    if config_updates:
        update_config(**config_updates)
        logger.info(f"Updated configuration: {config_updates}")
    
    # Log current configuration
    config = get_config()
    logger.info(f"Starting FastServe with configuration:")
    logger.info(f"  Model: {config.model_name}")
    logger.info(f"  Device: {config.device}")
    logger.info(f"  Max sequence length: {config.max_sequence_length}")
    logger.info(f"  Number of queues: {config.num_priority_queues}")
    logger.info(f"  Preemption enabled: {config.preemption_enabled}")
    
    # Start server
    try:
        run_server(
            host=args.host,
            port=args.port,
            workers=args.workers,
            reload=args.reload,
            log_level=args.log_level.lower()
        )
    except KeyboardInterrupt:
        logger.info("Shutting down FastServe server...")
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
