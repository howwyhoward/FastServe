"""
Key-Value Cache Manager for FastServe
Manages distributed key-value caches for Transformer models
"""

import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Any
import torch
from dataclasses import dataclass

from .models import InferenceJob, CacheEntry
from .memory_manager import MemoryManager
from .config import get_config

logger = logging.getLogger(__name__)


@dataclass
class LayerCache:
    """Cache for a single transformer layer"""
    layer_id: int
    key_cache: Dict[str, torch.Tensor]
    value_cache: Dict[str, torch.Tensor]
    sequence_lengths: Dict[str, int]
    
    def __post_init__(self):
        if not hasattr(self, 'key_cache'):
            self.key_cache = {}
        if not hasattr(self, 'value_cache'):
            self.value_cache = {}
        if not hasattr(self, 'sequence_lengths'):
            self.sequence_lengths = {}


class KeyValueCacheManager:
    """
    Manages key-value caches for Transformer models
    
    Key features:
    - Layer-wise cache management
    - Integration with memory manager for GPU/CPU swapping
    - Support for incremental cache updates
    - Cache invalidation and cleanup
    """
    
    def __init__(self, memory_manager: MemoryManager, config=None):
        self.config = config or get_config()
        self.memory_manager = memory_manager
        self.device = self.config.device
        
        # Cache storage
        self.layer_caches: Dict[int, LayerCache] = {}
        self.job_cache_keys: Dict[str, List[str]] = {}  # Track cache keys per job
        
        # Statistics
        self.cache_hits: int = 0
        self.cache_misses: int = 0
        self.total_cache_operations: int = 0
        
        logger.info("Key-Value Cache Manager initialized")
    
    async def get_cache(self, job_id: str, layer_id: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Retrieve key-value cache for a specific job and layer
        """
        self.total_cache_operations += 1
        
        # Try to get from memory manager first
        cache_entry = await self.memory_manager.get_cache_entry(job_id, layer_id)
        
        if cache_entry:
            self.cache_hits += 1
            logger.debug(f"Cache hit for job {job_id}, layer {layer_id}")
            return cache_entry.key_tensor, cache_entry.value_tensor
        
        self.cache_misses += 1
        logger.debug(f"Cache miss for job {job_id}, layer {layer_id}")
        return None
    
    async def store_cache(self, job_id: str, layer_id: int, 
                         key_tensor: torch.Tensor, value_tensor: torch.Tensor) -> bool:
        """
        Store key-value cache for a specific job and layer
        """
        # Ensure tensors are on the correct device
        if key_tensor.device.type != self.device:
            key_tensor = key_tensor.to(self.device)
        if value_tensor.device.type != self.device:
            value_tensor = value_tensor.to(self.device)
        
        # Store in memory manager
        success = await self.memory_manager.allocate_cache_entry(
            job_id, layer_id, key_tensor, value_tensor
        )
        
        if success:
            # Track cache keys for this job
            if job_id not in self.job_cache_keys:
                self.job_cache_keys[job_id] = []
            
            cache_key = f"{job_id}_{layer_id}"
            if cache_key not in self.job_cache_keys[job_id]:
                self.job_cache_keys[job_id].append(cache_key)
            
            logger.debug(f"Stored cache for job {job_id}, layer {layer_id}")
        
        return success
    
    async def update_cache(self, job_id: str, layer_id: int,
                          new_key_tensor: torch.Tensor, new_value_tensor: torch.Tensor) -> bool:
        """
        Update existing cache with new key-value tensors (incremental update)
        """
        # Get existing cache
        existing_cache = await self.get_cache(job_id, layer_id)
        
        if existing_cache is None:
            # No existing cache, create new one
            return await self.store_cache(job_id, layer_id, new_key_tensor, new_value_tensor)
        
        existing_key, existing_value = existing_cache
        
        # Concatenate new tensors with existing ones
        # Assuming sequence dimension is -2 (typical for transformer caches)
        updated_key = torch.cat([existing_key, new_key_tensor], dim=-2)
        updated_value = torch.cat([existing_value, new_value_tensor], dim=-2)
        
        # Remove old cache entry
        await self.memory_manager.remove_cache_entry(job_id, layer_id)
        
        # Store updated cache
        return await self.store_cache(job_id, layer_id, updated_key, updated_value)
    
    async def invalidate_job_cache(self, job_id: str) -> None:
        """
        Remove all cache entries for a specific job
        """
        if job_id in self.job_cache_keys:
            # Remove from memory manager
            await self.memory_manager.remove_cache_entry(job_id)
            
            # Clean up tracking
            del self.job_cache_keys[job_id]
            
            logger.debug(f"Invalidated all cache for job {job_id}")
    
    async def get_cache_for_generation(self, job: InferenceJob, 
                                     num_layers: int) -> Optional[Tuple[torch.Tensor, ...]]:
        """
        Get complete cache state for text generation
        Returns past_key_values in the format expected by transformers
        """
        if not job.past_key_values:
            return None
        
        past_key_values = []
        
        for layer_id in range(num_layers):
            cache_result = await self.get_cache(job.job_id, layer_id)
            
            if cache_result is None:
                # Missing cache for this layer, return None to regenerate
                return None
            
            key_tensor, value_tensor = cache_result
            past_key_values.append((key_tensor, value_tensor))
        
        return tuple(past_key_values)
    
    async def store_cache_from_generation(self, job: InferenceJob, 
                                        past_key_values: Tuple[Tuple[torch.Tensor, torch.Tensor], ...]) -> bool:
        """
        Store cache state from text generation output
        """
        if not past_key_values:
            return False
        
        success = True
        
        for layer_id, (key_tensor, value_tensor) in enumerate(past_key_values):
            layer_success = await self.store_cache(job.job_id, layer_id, key_tensor, value_tensor)
            success = success and layer_success
        
        # Update job's cache state
        job.past_key_values = past_key_values
        
        return success
    
    async def update_cache_from_generation(self, job: InferenceJob,
                                         new_past_key_values: Tuple[Tuple[torch.Tensor, torch.Tensor], ...]) -> bool:
        """
        Update cache state with new tokens from generation
        """
        if not new_past_key_values:
            return False
        
        # If no existing cache, store as new
        if not job.past_key_values:
            return await self.store_cache_from_generation(job, new_past_key_values)
        
        success = True
        
        for layer_id, (new_key, new_value) in enumerate(new_past_key_values):
            # Extract only the new tokens (last token's key/value)
            new_key_token = new_key[..., -1:, :]  # Last token only
            new_value_token = new_value[..., -1:, :]
            
            layer_success = await self.update_cache(job.job_id, layer_id, new_key_token, new_value_token)
            success = success and layer_success
        
        # Update job's cache state
        job.past_key_values = new_past_key_values
        
        return success
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        hit_rate = (self.cache_hits / max(1, self.total_cache_operations)) * 100
        
        return {
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'total_operations': self.total_cache_operations,
            'hit_rate_percent': hit_rate,
            'active_jobs': len(self.job_cache_keys),
            'total_cache_keys': sum(len(keys) for keys in self.job_cache_keys.values())
        }
    
    async def cleanup_expired_caches(self, max_age_seconds: float = 3600) -> int:
        """
        Clean up caches for jobs that are too old
        Returns number of cache entries cleaned up
        """
        import time
        current_time = time.time()
        expired_jobs = []
        
        # This is a simplified cleanup - in a real implementation,
        # you'd track job timestamps and clean up accordingly
        for job_id in list(self.job_cache_keys.keys()):
            # For now, just clean up jobs with more than 100 cache entries
            # In practice, you'd use actual timestamp tracking
            if len(self.job_cache_keys[job_id]) > 100:
                expired_jobs.append(job_id)
        
        cleaned_count = 0
        for job_id in expired_jobs:
            await self.invalidate_job_cache(job_id)
            cleaned_count += len(self.job_cache_keys.get(job_id, []))
        
        if cleaned_count > 0:
            logger.info(f"Cleaned up {cleaned_count} expired cache entries")
        
        return cleaned_count
    
    async def preload_cache(self, job: InferenceJob, model_output: Any) -> bool:
        """
        Preload cache from initial model forward pass
        """
        if hasattr(model_output, 'past_key_values') and model_output.past_key_values:
            return await self.store_cache_from_generation(job, model_output.past_key_values)
        return False
    
    async def get_memory_pressure(self) -> float:
        """
        Get current memory pressure as a value between 0 and 1
        """
        stats = self.memory_manager.get_memory_stats()
        return stats.gpu_utilization / 100.0
    
    async def start(self) -> None:
        """Start cache manager (if any background tasks needed)"""
        logger.info("Key-Value Cache Manager started")
    
    async def stop(self) -> None:
        """Stop cache manager and clean up"""
        # Clean up all caches
        for job_id in list(self.job_cache_keys.keys()):
            await self.invalidate_job_cache(job_id)
        
        logger.info("Key-Value Cache Manager stopped")
