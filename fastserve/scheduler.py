"""
Skip-Join Multi-Level Feedback Queue (MLFQ) Scheduler
Implements the core scheduling algorithm from FastServe
"""

import asyncio
import time
import logging
from typing import List, Dict, Optional, Tuple
from collections import deque
from dataclasses import dataclass, field

from .models import InferenceJob, JobStatus, Priority
from .config import get_config
from .job_profiler import JobProfiler, JobProfile

logger = logging.getLogger(__name__)


@dataclass
class PriorityQueue:
    """A priority queue with skip-join logic"""
    
    queue_id: int
    priority: Priority
    jobs: deque = field(default_factory=deque)
    time_quantum: float = 1.0  # Time slice for round-robin within queue
    max_capacity: int = 100
    
    def __len__(self) -> int:
        return len(self.jobs)
    
    def is_empty(self) -> bool:
        return len(self.jobs) == 0
    
    def is_full(self) -> bool:
        return len(self.jobs) >= self.max_capacity
    
    def enqueue(self, job: InferenceJob) -> bool:
        """Add job to queue if not full"""
        if self.is_full():
            return False
        
        job.queue_id = self.queue_id
        job.priority = self.priority
        self.jobs.append(job)
        logger.debug(f"Job {job.job_id} added to queue {self.queue_id}")
        return True
    
    def dequeue(self) -> Optional[InferenceJob]:
        """Remove and return next job"""
        if self.is_empty():
            return None
        
        job = self.jobs.popleft()
        logger.debug(f"Job {job.job_id} dequeued from queue {self.queue_id}")
        return job
    
    def peek(self) -> Optional[InferenceJob]:
        """Return next job without removing it"""
        return self.jobs[0] if not self.is_empty() else None


class SkipJoinMLFQScheduler:
    """
    Skip-Join Multi-Level Feedback Queue Scheduler
    
    Key features:
    - Multiple priority queues with different time quantums
    - Skip-join logic: longer jobs skip to lower priority queues
    - Preemptive scheduling with token-level granularity
    - Promotion/demotion based on execution behavior
    """
    
    def __init__(self, config=None):
        self.config = config or get_config()
        
        # Initialize Job Profiler
        self.job_profiler = JobProfiler(self.config)
        
        # Scheduling state
        self.current_job: Optional[InferenceJob] = None
        self.current_queue_id: int = 0
        self.quantum_start_time: float = 0.0
        
        # Statistics
        self.total_jobs_processed: int = 0
        self.total_preemptions: int = 0
        self.queue_stats: Dict[int, Dict[str, float]] = {}
        
        # Initialize priority queues (after stats are initialized)
        self.queues: List[PriorityQueue] = []
        self._init_queues()
        
        # Control
        self.is_running: bool = False
        self._scheduler_task: Optional[asyncio.Task] = None
        
        logger.info(f"Initialized Skip-Join MLFQ with {len(self.queues)} queues and Job Profiler")
    
    def _init_queues(self) -> None:
        """Initialize the priority queues with different time quantums"""
        priorities = [Priority.HIGH, Priority.MEDIUM, Priority.LOW, Priority.BACKGROUND]
        time_quantums = [0.5, 1.0, 2.0, 4.0]  # Increasing time quantum per queue
        
        for i in range(self.config.num_priority_queues):
            queue = PriorityQueue(
                queue_id=i,
                priority=priorities[i] if i < len(priorities) else Priority.BACKGROUND,
                time_quantum=time_quantums[i] if i < len(time_quantums) else 4.0,
                max_capacity=self.config.queue_capacity
            )
            self.queues.append(queue)
            
            # Initialize stats
            self.queue_stats[i] = {
                'jobs_processed': 0,
                'total_wait_time': 0.0,
                'total_execution_time': 0.0,
                'preemptions': 0
            }
    
    async def submit_job(self, job: InferenceJob) -> bool:
        """
        Submit a new job to the scheduler with intelligent profiling and skip-join logic
        """
        # Profile the job to predict its characteristics
        profile = self.job_profiler.profile_job(job)
        
        # Determine optimal initial queue based on job profile
        target_queue_id = self.job_profiler.get_optimal_initial_queue(job, profile)
        
        # Store profile for later reference
        job.profile = profile
        
        # Try to place job in target queue or next available queue
        for queue_id in range(target_queue_id, len(self.queues)):
            if self.queues[queue_id].enqueue(job):
                job.queue_time = time.time() - job.created_at
                logger.info(f"Job {job.job_id} submitted to queue {queue_id} "
                          f"(complexity: {profile.complexity.name}, "
                          f"predicted_duration: {profile.predicted_duration:.2f}s)")
                return True
        
        # All queues are full
        logger.warning(f"All queues full, rejecting job {job.job_id}")
        return False
    
    def update_profiler_on_completion(self, job: InferenceJob) -> None:
        """
        Update job profiler with completion data for learning
        """
        if hasattr(job, 'profile') and job.profile:
            actual_execution_time = job.get_execution_time() or 0.0
            actual_tokens = job.total_tokens_generated
            
            self.job_profiler.update_with_completion_data(
                job, actual_execution_time, actual_tokens
            )
            
            logger.debug(f"Updated profiler with completion data for job {job.job_id}: "
                        f"execution_time={actual_execution_time:.2f}s, "
                        f"tokens={actual_tokens}")
        else:
            logger.warning(f"Job {job.job_id} completed without profile data")
    
    async def get_next_job(self) -> Optional[InferenceJob]:
        """Get the next job to execute using MLFQ policy"""
        
        # Check queues in priority order (highest priority first)
        for queue in self.queues:
            if not queue.is_empty():
                job = queue.dequeue()
                if job:
                    # Skip jobs that are already marked as failed to prevent infinite loops
                    if job.status == JobStatus.FAILED:
                        logger.debug(f"Skipping failed job {job.job_id}")
                        continue
                    
                    # Check retry limit to prevent infinite retry loops
                    if job.retry_count >= job.max_retries:
                        job.update_status(JobStatus.FAILED)
                        logger.warning(f"Job {job.job_id} exceeded retry limit ({job.max_retries}), marking as failed")
                        continue
                        
                    job.update_status(JobStatus.RUNNING)
                    self.current_job = job
                    self.current_queue_id = queue.queue_id
                    self.quantum_start_time = time.time()
                    
                    # Update statistics
                    self.queue_stats[queue.queue_id]['jobs_processed'] += 1
                    
                    logger.debug(f"Selected job {job.job_id} from queue {queue.queue_id} (retry {job.retry_count})")
                    return job
        
        return None
    
    def should_preempt_current_job(self) -> bool:
        """
        Determine if current job should be preempted
        
        Preemption occurs when:
        1. Time quantum exceeded
        2. Higher priority job arrives
        3. Memory pressure requires job swapping
        """
        if not self.current_job or not self.config.preemption_enabled:
            return False
        
        current_time = time.time()
        execution_time = current_time - self.quantum_start_time
        current_queue = self.queues[self.current_queue_id]
        
        # Check time quantum
        if execution_time >= current_queue.time_quantum:
            logger.debug(f"Time quantum exceeded for job {self.current_job.job_id}")
            return True
        
        # Check for higher priority jobs
        for i in range(self.current_queue_id):
            if not self.queues[i].is_empty():
                logger.debug(f"Higher priority job available, preempting {self.current_job.job_id}")
                return True
        
        return False
    
    async def preempt_current_job(self) -> None:
        """Preempt the currently running job"""
        if not self.current_job:
            return
        
        job = self.current_job
        job.update_status(JobStatus.PREEMPTED)
        
        # Increment retry count when requeuing
        job.retry_count += 1
        
        # Check if job has exceeded retry limit
        if job.retry_count >= job.max_retries:
            job.update_status(JobStatus.FAILED)
            logger.warning(f"Job {job.job_id} exceeded retry limit ({job.max_retries}), marking as failed")
            self.current_job = None
            self.current_queue_id = 0
            return
        
        # Determine where to place the preempted job
        target_queue_id = self._get_demotion_queue(job)
        
        # Place job back in appropriate queue
        if self.queues[target_queue_id].enqueue(job):
            logger.info(f"Job {job.job_id} preempted and moved to queue {target_queue_id} (retry {job.retry_count})")
            
            # Update statistics
            self.total_preemptions += 1
            self.queue_stats[self.current_queue_id]['preemptions'] += 1
        else:
            # Queue is full, mark job as failed
            job.update_status(JobStatus.FAILED)
            logger.error(f"Failed to requeue preempted job {job.job_id}")
        
        self.current_job = None
        self.current_queue_id = 0
    
    def _get_demotion_queue(self, job: InferenceJob) -> int:
        """Determine target queue for demoted job"""
        current_queue_id = job.queue_id
        
        # Promote if job has generated few tokens (finished quickly)
        if job.total_tokens_generated <= self.config.promotion_threshold:
            return max(0, current_queue_id - 1)
        
        # Demote if job has generated many tokens (long-running)
        elif job.total_tokens_generated >= self.config.demotion_threshold:
            return min(len(self.queues) - 1, current_queue_id + 1)
        
        # Stay in same queue
        return current_queue_id
    
    def get_profiler_statistics(self) -> Dict[str, float]:
        """Get job profiler performance statistics"""
        return self.job_profiler.get_statistics()
    
    def complete_current_job(self) -> None:
        """Mark current job as completed"""
        if self.current_job:
            self.current_job.update_status(JobStatus.COMPLETED)
            self.total_jobs_processed += 1
            
            # Update statistics
            queue_id = self.current_queue_id
            execution_time = time.time() - self.quantum_start_time
            self.queue_stats[queue_id]['total_execution_time'] += execution_time
            
            logger.info(f"Job {self.current_job.job_id} completed")
            self.current_job = None
            self.current_queue_id = 0
    
    def get_queue_stats(self) -> Dict[int, Dict[str, float]]:
        """Get scheduling statistics"""
        stats = {}
        for queue_id, queue_stats in self.queue_stats.items():
            queue = self.queues[queue_id]
            stats[queue_id] = {
                'current_length': len(queue),
                'jobs_processed': queue_stats['jobs_processed'],
                'avg_wait_time': (
                    queue_stats['total_wait_time'] / max(1, queue_stats['jobs_processed'])
                ),
                'avg_execution_time': (
                    queue_stats['total_execution_time'] / max(1, queue_stats['jobs_processed'])
                ),
                'preemptions': queue_stats['preemptions'],
                'priority': queue.priority.name,
                'time_quantum': queue.time_quantum
            }
        
        return stats
    
    async def start(self) -> None:
        """Start the scheduler"""
        self.is_running = True
        logger.info("Skip-Join MLFQ Scheduler started")
    
    async def stop(self) -> None:
        """Stop the scheduler"""
        self.is_running = False
        if self._scheduler_task:
            self._scheduler_task.cancel()
        logger.info("Skip-Join MLFQ Scheduler stopped")
