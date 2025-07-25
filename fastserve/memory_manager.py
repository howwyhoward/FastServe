"""
Memory Manager for FastServe
Handles GPU memory allocation, cache swapping, and memory pressure management
"""

import asyncio
import logging
import psutil
import torch
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
import time

from .models import InferenceJob, CacheEntry
from .config import get_config

logger = logging.getLogger(__name__)


@dataclass
class MemoryStats:
    """Memory usage statistics"""
    gpu_total_mb: float
    gpu_used_mb: float
    gpu_free_mb: float
    gpu_utilization: float
    cpu_total_mb: float
    cpu_used_mb: float
    cpu_utilization: float
    cache_entries_gpu: int
    cache_entries_cpu: int
    total_cache_size_mb: float


class MemoryManager:
    """
    Manages GPU and CPU memory for FastServe
    
    Key responsibilities:
    - Monitor GPU memory usage
    - Swap cache entries between GPU and CPU
    - Implement LRU eviction policy
    - Handle memory pressure situations
    """
    
    def __init__(self, config=None):
        self.config = config or get_config()
        self.device = self.config.device
        
        # Memory tracking
        self.gpu_cache_entries: Dict[str, CacheEntry] = {}
        self.cpu_cache_entries: Dict[str, CacheEntry] = {}
        self.gpu_memory_used: float = 0.0
        self.total_cache_size: float = 0.0
        
        # LRU tracking
        self.access_order: List[str] = []  # Most recent first
        
        # Memory thresholds
        self.gpu_threshold = self.config.gpu_memory_threshold
        self.max_cache_size = self.config.max_cache_size_mb * 1024 * 1024  # Convert to bytes
        
        # Async management
        self._memory_monitor_task: Optional[asyncio.Task] = None
        self.is_running: bool = False
        
        logger.info(f"Memory Manager initialized for device: {self.device}")
        logger.info(f"GPU threshold: {self.gpu_threshold}, Max cache: {self.config.max_cache_size_mb}MB")
    
    async def allocate_cache_entry(self, job_id: str, layer_id: int, 
                                 key_tensor: torch.Tensor, value_tensor: torch.Tensor) -> bool:
        """
        Allocate a new cache entry, managing memory pressure
        """
        cache_key = f"{job_id}_{layer_id}"
        
        # Create cache entry
        cache_entry = CacheEntry(
            job_id=job_id,
            layer_id=layer_id,
            key_tensor=key_tensor.clone(),
            value_tensor=value_tensor.clone(),
            sequence_length=key_tensor.size(-2)
        )
        
        # Check if we need to free memory first
        await self._ensure_memory_available(cache_entry.size_bytes)
        
        # Try to place on GPU first
        if self._can_place_on_gpu(cache_entry.size_bytes):
            self.gpu_cache_entries[cache_key] = cache_entry
            self.gpu_memory_used += cache_entry.size_bytes
            logger.debug(f"Allocated cache entry {cache_key} on GPU ({cache_entry.size_bytes} bytes)")
        else:
            # Place on CPU
            cache_entry.to_cpu()
            self.cpu_cache_entries[cache_key] = cache_entry
            logger.debug(f"Allocated cache entry {cache_key} on CPU ({cache_entry.size_bytes} bytes)")
        
        # Update access order
        self._update_access_order(cache_key)
        self.total_cache_size += cache_entry.size_bytes
        
        return True
    
    async def get_cache_entry(self, job_id: str, layer_id: int) -> Optional[CacheEntry]:
        """
        Retrieve a cache entry, potentially moving it to GPU
        """
        cache_key = f"{job_id}_{layer_id}"
        
        # Check GPU cache first
        if cache_key in self.gpu_cache_entries:
            entry = self.gpu_cache_entries[cache_key]
            self._update_access_order(cache_key)
            return entry
        
        # Check CPU cache
        if cache_key in self.cpu_cache_entries:
            entry = self.cpu_cache_entries[cache_key]
            
            # Try to move to GPU if there's space
            if self._can_place_on_gpu(entry.size_bytes):
                await self._move_to_gpu(cache_key, entry)
            
            self._update_access_order(cache_key)
            return entry
        
        return None
    
    async def remove_cache_entry(self, job_id: str, layer_id: int = None) -> None:
        """
        Remove cache entries for a job (all layers if layer_id is None)
        """
        if layer_id is not None:
            # Remove specific layer
            cache_key = f"{job_id}_{layer_id}"
            await self._remove_single_entry(cache_key)
        else:
            # Remove all entries for job
            keys_to_remove = [
                key for key in list(self.gpu_cache_entries.keys()) + list(self.cpu_cache_entries.keys())
                if key.startswith(f"{job_id}_")
            ]
            
            for key in keys_to_remove:
                await self._remove_single_entry(key)
        
        logger.debug(f"Removed cache entries for job {job_id}")
    
    async def _remove_single_entry(self, cache_key: str) -> None:
        """Remove a single cache entry"""
        entry = None
        
        if cache_key in self.gpu_cache_entries:
            entry = self.gpu_cache_entries.pop(cache_key)
            self.gpu_memory_used -= entry.size_bytes
        elif cache_key in self.cpu_cache_entries:
            entry = self.cpu_cache_entries.pop(cache_key)
        
        if entry:
            self.total_cache_size -= entry.size_bytes
            if cache_key in self.access_order:
                self.access_order.remove(cache_key)
    
    def _can_place_on_gpu(self, size_bytes: float) -> bool:
        """Check if entry can be placed on GPU without exceeding threshold"""
        if self.device == "cpu":
            return False
        
        try:
            if self.device == "mps":
                # For MPS, we use a heuristic based on configured limits
                estimated_usage = (self.gpu_memory_used + size_bytes) / (1024 * 1024 * 1024)  # GB
                return estimated_usage < (self.config.max_cache_size_mb / 1024)
            else:
                # For CUDA
                total_memory = torch.cuda.get_device_properties(0).total_memory
                current_memory = torch.cuda.memory_allocated(0)
                available_memory = total_memory - current_memory
                return size_bytes < available_memory * self.gpu_threshold
        except Exception as e:
            logger.warning(f"Error checking GPU memory: {e}")
            return False
    
    async def _ensure_memory_available(self, required_bytes: float) -> None:
        """Ensure enough memory is available by evicting entries if necessary"""
        # Check total cache size limit
        while (self.total_cache_size + required_bytes) > self.max_cache_size:
            if not await self._evict_lru_entry():
                break
        
        # Check GPU memory pressure
        if self.device != "cpu":
            while not self._can_place_on_gpu(required_bytes) and self.gpu_cache_entries:
                if not await self._move_lru_to_cpu():
                    break
    
    async def _evict_lru_entry(self) -> bool:
        """Evict least recently used cache entry"""
        if not self.access_order:
            return False
        
        # Find LRU entry (last in access_order)
        lru_key = self.access_order[-1]
        await self._remove_single_entry(lru_key)
        logger.debug(f"Evicted LRU cache entry: {lru_key}")
        return True
    
    async def _move_lru_to_cpu(self) -> bool:
        """Move least recently used GPU entry to CPU"""
        if not self.gpu_cache_entries:
            return False
        
        # Find LRU GPU entry
        lru_gpu_key = None
        for key in reversed(self.access_order):
            if key in self.gpu_cache_entries:
                lru_gpu_key = key
                break
        
        if not lru_gpu_key:
            return False
        
        entry = self.gpu_cache_entries.pop(lru_gpu_key)
        self.gpu_memory_used -= entry.size_bytes
        
        # Move to CPU
        entry.to_cpu()
        self.cpu_cache_entries[lru_gpu_key] = entry
        
        logger.debug(f"Moved cache entry {lru_gpu_key} from GPU to CPU")
        return True
    
    async def _move_to_gpu(self, cache_key: str, entry: CacheEntry) -> bool:
        """Move cache entry from CPU to GPU"""
        if cache_key not in self.cpu_cache_entries:
            return False
        
        # Ensure space available
        await self._ensure_memory_available(entry.size_bytes)
        
        if self._can_place_on_gpu(entry.size_bytes):
            # Remove from CPU cache
            self.cpu_cache_entries.pop(cache_key)
            
            # Move to GPU
            entry.to_gpu(self.device)
            self.gpu_cache_entries[cache_key] = entry
            self.gpu_memory_used += entry.size_bytes
            
            logger.debug(f"Moved cache entry {cache_key} from CPU to GPU")
            return True
        
        return False
    
    def _update_access_order(self, cache_key: str) -> None:
        """Update LRU access order"""
        if cache_key in self.access_order:
            self.access_order.remove(cache_key)
        self.access_order.insert(0, cache_key)  # Most recent first
        
        # Limit access order size to prevent memory issues
        if len(self.access_order) > 10000:
            self.access_order = self.access_order[:5000]
    
    def get_memory_stats(self) -> MemoryStats:
        """Get current memory usage statistics"""
        # CPU memory
        cpu_memory = psutil.virtual_memory()
        cpu_total_mb = cpu_memory.total / (1024 * 1024)
        cpu_used_mb = cpu_memory.used / (1024 * 1024)
        cpu_utilization = cpu_memory.percent
        
        # GPU memory (approximated for MPS)
        gpu_total_mb = self.config.max_cache_size_mb
        gpu_used_mb = self.gpu_memory_used / (1024 * 1024)
        gpu_free_mb = gpu_total_mb - gpu_used_mb
        gpu_utilization = (gpu_used_mb / gpu_total_mb) * 100 if gpu_total_mb > 0 else 0
        
        return MemoryStats(
            gpu_total_mb=gpu_total_mb,
            gpu_used_mb=gpu_used_mb,
            gpu_free_mb=gpu_free_mb,
            gpu_utilization=gpu_utilization,
            cpu_total_mb=cpu_total_mb,
            cpu_used_mb=cpu_used_mb,
            cpu_utilization=cpu_utilization,
            cache_entries_gpu=len(self.gpu_cache_entries),
            cache_entries_cpu=len(self.cpu_cache_entries),
            total_cache_size_mb=self.total_cache_size / (1024 * 1024)
        )
    
    async def _memory_monitor_loop(self) -> None:
        """Background task to monitor memory usage"""
        while self.is_running:
            try:
                stats = self.get_memory_stats()
                
                # Log memory stats periodically
                if stats.gpu_utilization > 90:
                    logger.warning(f"High GPU memory usage: {stats.gpu_utilization:.1f}%")
                
                if stats.cpu_utilization > 90:
                    logger.warning(f"High CPU memory usage: {stats.cpu_utilization:.1f}%")
                
                # Proactive memory management
                if stats.gpu_utilization > self.gpu_threshold * 100:
                    await self._move_lru_to_cpu()
                
                await asyncio.sleep(5.0)  # Check every 5 seconds
                
            except Exception as e:
                logger.error(f"Error in memory monitor: {e}")
                await asyncio.sleep(10.0)
    
    async def start(self) -> None:
        """Start memory manager"""
        self.is_running = True
        self._memory_monitor_task = asyncio.create_task(self._memory_monitor_loop())
        logger.info("Memory Manager started")
    
    async def stop(self) -> None:
        """Stop memory manager"""
        self.is_running = False
        if self._memory_monitor_task:
            self._memory_monitor_task.cancel()
            try:
                await self._memory_monitor_task
            except asyncio.CancelledError:
                pass
        logger.info("Memory Manager stopped")
