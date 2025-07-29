"""
Data models for FastServe system
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Union
from enum import Enum
import time
import uuid
import random
import torch
from pydantic import BaseModel


class JobStatus(Enum):
    """Status of inference jobs"""
    PENDING = "pending"
    RUNNING = "running"
    PREEMPTED = "preempted"
    COMPLETED = "completed"
    FAILED = "failed"


class Priority(Enum):
    """Priority levels for jobs"""
    HIGH = 0
    MEDIUM = 1
    LOW = 2
    BACKGROUND = 3


@dataclass
class InferenceJob:
    """Represents an inference job in the system"""
    
    job_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    prompt: str = ""
    max_new_tokens: int = 100
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    do_sample: bool = True
    
    # Job tracking
    status: JobStatus = JobStatus.PENDING
    priority: Priority = Priority.MEDIUM
    queue_id: int = 1
    
    # Timing information
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    last_preempted_at: Optional[float] = None
    
    # Generation tracking
    generated_tokens: List[int] = field(default_factory=list)
    generated_text: str = ""
    current_position: int = 0
    input_length: int = 0
    
    # Key-value cache state
    past_key_values: Optional[Any] = None
    attention_mask: Optional[torch.Tensor] = None
    position_ids: Optional[torch.Tensor] = None
    
    # Performance metrics
    total_tokens_generated: int = 0
    preemption_count: int = 0
    queue_time: float = 0.0
    execution_time: float = 0.0
    retry_count: int = 0
    max_retries: int = 3
    
    # Job profiling data
    profile: Optional[Any] = None  # Will store JobProfile from job_profiler
    
    def __post_init__(self):
        """Initialize computed fields"""
        self.input_length = len(self.prompt.split()) if self.prompt else 0
        
    def get_completion_time(self) -> Optional[float]:
        """Get job completion time in seconds"""
        if self.completed_at and self.created_at:
            return self.completed_at - self.created_at
        return None
    
    def get_execution_time(self) -> Optional[float]:
        """Get actual execution time (excluding queue time)"""
        if self.started_at and self.completed_at:
            return self.completed_at - self.started_at
        return None
    
    def update_status(self, status: JobStatus) -> None:
        """Update job status with timestamp"""
        self.status = status
        current_time = time.time()
        
        if status == JobStatus.RUNNING:
            self.started_at = current_time
        elif status == JobStatus.COMPLETED:
            self.completed_at = current_time
        elif status == JobStatus.PREEMPTED:
            self.last_preempted_at = current_time
            self.preemption_count += 1
    
    def should_skip_queue(self, queue_id: int, skip_threshold: float) -> bool:
        """Determine if job should skip to a lower priority queue"""
        # Skip-join logic based on input length
        length_factor = min(self.input_length / 100.0, 1.0)  # Normalize to [0,1]
        skip_probability = length_factor * skip_threshold
        
        return random.random() < skip_probability and queue_id < 3


@dataclass 
class CacheEntry:
    """Represents a key-value cache entry"""
    
    job_id: str
    layer_id: int
    key_tensor: torch.Tensor
    value_tensor: torch.Tensor
    sequence_length: int
    last_accessed: float = field(default_factory=time.time)
    is_on_gpu: bool = True
    size_bytes: int = 0
    
    def __post_init__(self):
        """Calculate cache entry size"""
        self.size_bytes = (
            self.key_tensor.numel() * self.key_tensor.element_size() +
            self.value_tensor.numel() * self.value_tensor.element_size()
        )
    
    def to_cpu(self) -> None:
        """Move cache entry to CPU memory"""
        if self.is_on_gpu:
            self.key_tensor = self.key_tensor.cpu()
            self.value_tensor = self.value_tensor.cpu()
            self.is_on_gpu = False
    
    def to_gpu(self, device: str) -> None:
        """Move cache entry to GPU memory"""
        if not self.is_on_gpu:
            self.key_tensor = self.key_tensor.to(device)
            self.value_tensor = self.value_tensor.to(device)
            self.is_on_gpu = True


# Pydantic models for API
class GenerationRequest(BaseModel):
    """Request model for text generation"""
    prompt: str
    max_new_tokens: int = 100
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    do_sample: bool = True
    stream: bool = False


class GenerationResponse(BaseModel):
    """Response model for text generation"""
    job_id: str
    generated_text: str
    total_tokens: int
    completion_time: float
    queue_time: float
    execution_time: float
    preemption_count: int


class StreamResponse(BaseModel):
    """Streaming response model"""
    job_id: str
    token: str
    is_final: bool = False
    total_tokens: int = 0


class SystemStats(BaseModel):
    """System statistics"""
    active_jobs: int
    completed_jobs: int
    failed_jobs: int
    average_queue_time: float
    average_execution_time: float
    gpu_memory_used: float
    cpu_memory_used: float
    cache_hit_rate: float
