"""
Configuration settings for FastServe system
Optimized for Mac M2 Apple Silicon with MPS support
"""

import torch
from dataclasses import dataclass
from typing import Optional, List
import os


@dataclass
class FastServeConfig:
    """Main configuration for FastServe system"""
    
    # Model settings
    model_name: str = "gpt2"  # Test with downloaded GPT-2
    use_ollama: bool = False  # Test HuggingFace backend first
    ollama_model: str = "deepseek-r1:14b"  # Ollama model name
    max_sequence_length: int = 1024
    max_new_tokens: int = 256
    
    # Device settings (optimized for Mac M2)
    device: str = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype: torch.dtype = torch.float16 if torch.backends.mps.is_available() else torch.float32
    
    # Scheduler settings
    num_priority_queues: int = 4
    queue_capacity: int = 100
    preemption_enabled: bool = True
    
    # Memory management
    gpu_memory_threshold: float = 0.8  # Start offloading at 80% GPU memory
    max_cache_size_mb: int = 1024  # 1GB cache on Mac M2
    enable_memory_offloading: bool = True
    
    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000
    max_concurrent_requests: int = 10
    
    # Performance settings
    batch_size: int = 1  # Start with 1 for single requests
    prefill_batch_size: int = 4
    decode_batch_size: int = 8
    
    # Skip-Join MLFQ settings
    skip_threshold: float = 0.7  # Skip probability for longer sequences
    promotion_threshold: int = 5  # Tokens to generate before promotion
    demotion_threshold: int = 20  # Tokens before demotion
    
    @classmethod
    def get_optimal_config(cls) -> 'FastServeConfig':
        """Get optimal configuration for Mac M2"""
        config = cls()
        
        # Detect available memory and adjust settings
        if torch.backends.mps.is_available():
            # Mac M2 specific optimizations
            config.device = "mps"
            config.dtype = torch.float16
            config.max_cache_size_mb = 2048  # 2GB on M2
            config.max_concurrent_requests = 8
        else:
            # Fallback to CPU
            config.device = "cpu"
            config.dtype = torch.float32
            config.max_cache_size_mb = 512
            config.max_concurrent_requests = 4
            
        return config


# Global configuration instance
CONFIG = FastServeConfig.get_optimal_config()


def update_config(**kwargs) -> None:
    """Update global configuration"""
    global CONFIG
    for key, value in kwargs.items():
        if hasattr(CONFIG, key):
            setattr(CONFIG, key, value)
        else:
            raise ValueError(f"Unknown configuration key: {key}")


def get_config() -> FastServeConfig:
    """Get current global configuration"""
    return CONFIG
