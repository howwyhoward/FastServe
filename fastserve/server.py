"""
FastServe Server
Main FastAPI server that orchestrates all components
"""

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse, HTMLResponse
import json

from .models import (
    GenerationRequest, GenerationResponse, StreamResponse, 
    SystemStats, InferenceJob, JobStatus
)
from .scheduler import SkipJoinMLFQScheduler
from .memory_manager import MemoryManager
from .cache_manager import KeyValueCacheManager
from .inference_engine import InferenceEngine
from .config import get_config, update_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FastServeServer:
    """
    Main FastServe server that coordinates all components
    """
    
    def __init__(self, config=None):
        self.config = config or get_config()
        
        # Initialize components
        self.scheduler = SkipJoinMLFQScheduler(self.config)
        self.memory_manager = MemoryManager(self.config)
        self.cache_manager = KeyValueCacheManager(self.memory_manager, self.config)
        self.inference_engine = InferenceEngine(self.scheduler, self.cache_manager, self.config)
        
        # Set up job completion callback
        self.inference_engine.job_completion_callback = self._on_job_completion
        
        # Server state
        self.is_running = False
        self.active_jobs: Dict[str, InferenceJob] = {}
        self.completed_jobs: Dict[str, InferenceJob] = {}
        self.failed_jobs: Dict[str, InferenceJob] = {}
        
        # Background tasks
        self._inference_task: Optional[asyncio.Task] = None
        
        # Statistics
        self.start_time = time.time()
        
        logger.info("FastServe Server initialized")
    
    def _on_job_completion(self, job: InferenceJob, success: bool) -> None:
        """Callback for when jobs complete or fail"""
        job_id = job.job_id
        
        if job_id in self.active_jobs:
            del self.active_jobs[job_id]
            
            if success and job.status == JobStatus.COMPLETED:
                # Update job profiler with completion data for learning
                self.scheduler.update_profiler_on_completion(job)
                
                self.completed_jobs[job_id] = job
                logger.info(f"Job {job_id} completed successfully")
            else:
                self.failed_jobs[job_id] = job
                logger.warning(f"Job {job_id} failed or was terminated")
    
    async def start_all_components(self) -> None:
        """Start all server components"""
        try:
            # Start components in dependency order
            await self.memory_manager.start()
            await self.cache_manager.start()
            await self.scheduler.start()
            await self.inference_engine.start()
            
            # Start inference loop
            self._inference_task = asyncio.create_task(
                self.inference_engine.run_inference_loop()
            )
            
            self.is_running = True
            logger.info("All FastServe components started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start components: {e}")
            await self.stop_all_components()
            raise
    
    async def stop_all_components(self) -> None:
        """Stop all server components"""
        self.is_running = False
        
        # Stop inference loop
        if self._inference_task:
            self._inference_task.cancel()
            try:
                await self._inference_task
            except asyncio.CancelledError:
                pass
        
        # Stop components in reverse order
        await self.inference_engine.stop()
        await self.scheduler.stop()
        await self.cache_manager.stop()
        await self.memory_manager.stop()
        
        logger.info("All FastServe components stopped")
    
    async def submit_generation_request(self, request: GenerationRequest) -> str:
        """Submit a new generation request and return job ID"""
        
        # Create inference job
        job = InferenceJob(
            prompt=request.prompt,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            do_sample=request.do_sample
        )
        
        # Submit to scheduler
        success = await self.scheduler.submit_job(job)
        
        if not success:
            raise HTTPException(status_code=503, detail="Server overloaded, please try again later")
        
        # Track job
        self.active_jobs[job.job_id] = job
        
        logger.info(f"Submitted job {job.job_id} with prompt length {len(request.prompt)}")
        return job.job_id
    
    async def get_generation_result(self, job_id: str) -> GenerationResponse:
        """Get result for a completed generation job"""
        
        # Check completed jobs
        if job_id in self.completed_jobs:
            job = self.completed_jobs[job_id]
            logger.debug(f"Job {job_id} found in completed jobs")
            return GenerationResponse(
                job_id=job.job_id,
                generated_text=job.generated_text,
                total_tokens=job.total_tokens_generated,
                completion_time=job.get_completion_time() or 0.0,
                queue_time=job.queue_time,
                execution_time=job.get_execution_time() or 0.0,
                preemption_count=job.preemption_count
            )
        
        # Check failed jobs
        if job_id in self.failed_jobs:
            job = self.failed_jobs[job_id]
            error_msg = f"Job failed during processing. Status: {job.status.value}, Retries: {job.retry_count}"
            logger.error(f"Job {job_id} failed: {error_msg}")
            raise HTTPException(status_code=500, detail=error_msg)
        
        # Check active jobs
        if job_id in self.active_jobs:
            job = self.active_jobs[job_id]
            logger.debug(f"Job {job_id} status: {job.status.value}, tokens: {job.total_tokens_generated}")
            
            if job.status == JobStatus.COMPLETED:
                # Move to completed jobs
                self.completed_jobs[job_id] = job
                del self.active_jobs[job_id]
                return await self.get_generation_result(job_id)
            elif job.status == JobStatus.FAILED:
                # Move to failed jobs
                self.failed_jobs[job_id] = job
                del self.active_jobs[job_id]
                return await self.get_generation_result(job_id)
            else:
                # Still processing
                raise HTTPException(status_code=202, detail=f"Job still processing (status: {job.status.value})")
        
        # Job not found anywhere
        raise HTTPException(status_code=404, detail="Job not found")
    
    async def generate_text_streaming(self, request: GenerationRequest):
        """Generate text with streaming response"""
        
        # Create inference job
        job = InferenceJob(
            prompt=request.prompt,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            do_sample=request.do_sample
        )
        
        # For streaming, we bypass the scheduler and generate directly
        # This is a simplified approach - in production, you'd integrate streaming with scheduling
        try:
            async def generate_stream():
                token_count = 0
                async for token in self.inference_engine.generate_streaming(job):
                    token_count += 1
                    response = StreamResponse(
                        job_id=job.job_id,
                        token=token,
                        is_final=False,
                        total_tokens=token_count
                    )
                    yield f"data: {response.model_dump_json()}\n\n"
                
                # Send final message
                final_response = StreamResponse(
                    job_id=job.job_id,
                    token="",
                    is_final=True,
                    total_tokens=token_count
                )
                yield f"data: {final_response.model_dump_json()}\n\n"
            
            return StreamingResponse(
                generate_stream(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                }
            )
            
        except Exception as e:
            logger.error(f"Streaming generation failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get status of a specific job"""
        
        # Check all job collections
        for jobs_dict, status in [
            (self.active_jobs, "active"),
            (self.completed_jobs, "completed"),
            (self.failed_jobs, "failed")
        ]:
            if job_id in jobs_dict:
                job = jobs_dict[job_id]
                return {
                    "job_id": job.job_id,
                    "status": job.status.value,
                    "collection": status,
                    "created_at": job.created_at,
                    "started_at": job.started_at,
                    "completed_at": job.completed_at,
                    "generated_tokens": job.total_tokens_generated,
                    "generated_text": job.generated_text[:100] + "..." if len(job.generated_text) > 100 else job.generated_text,
                    "preemption_count": job.preemption_count,
                    "queue_id": job.queue_id,
                    "priority": job.priority.name
                }
        
        return {"error": "Job not found"}
    
    def get_system_stats(self) -> SystemStats:
        """Get comprehensive system statistics"""
        
        # Calculate averages
        all_completed = list(self.completed_jobs.values())
        avg_queue_time = sum(job.queue_time for job in all_completed) / max(1, len(all_completed))
        avg_execution_time = sum(job.get_execution_time() or 0 for job in all_completed) / max(1, len(all_completed))
        
        # Get memory stats
        memory_stats = self.memory_manager.get_memory_stats()
        
        # Get cache stats
        cache_stats = self.cache_manager.get_cache_stats()
        
        return SystemStats(
            active_jobs=len(self.active_jobs),
            completed_jobs=len(self.completed_jobs),
            failed_jobs=len(self.failed_jobs),
            average_queue_time=avg_queue_time,
            average_execution_time=avg_execution_time,
            gpu_memory_used=memory_stats.gpu_utilization,
            cpu_memory_used=memory_stats.cpu_utilization,
            cache_hit_rate=cache_stats.get('hit_rate_percent', 0.0)
        )
    
    async def cleanup_old_jobs(self, max_age_hours: float = 24) -> int:
        """Clean up old completed and failed jobs"""
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        
        cleaned_count = 0
        
        # Clean completed jobs
        expired_completed = [
            job_id for job_id, job in self.completed_jobs.items()
            if (current_time - (job.completed_at or job.created_at)) > max_age_seconds
        ]
        
        for job_id in expired_completed:
            del self.completed_jobs[job_id]
            cleaned_count += 1
        
        # Clean failed jobs
        expired_failed = [
            job_id for job_id, job in self.failed_jobs.items()
            if (current_time - job.created_at) > max_age_seconds
        ]
        
        for job_id in expired_failed:
            del self.failed_jobs[job_id]
            cleaned_count += 1
        
        if cleaned_count > 0:
            logger.info(f"Cleaned up {cleaned_count} old jobs")
        
        return cleaned_count


# Create global server instance
server_instance = FastServeServer()


# FastAPI app with lifespan management
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage FastServe server lifecycle"""
    # Startup
    logger.info("Starting FastServe server...")
    await server_instance.start_all_components()
    
    yield
    
    # Shutdown
    logger.info("Shutting down FastServe server...")
    await server_instance.stop_all_components()


# Create FastAPI app
app = FastAPI(
    title="FastServe",
    description="Fast Distributed Inference Serving for Large Language Models",
    version="1.0.0",
    lifespan=lifespan
)


# API Routes
@app.post("/generate", response_model=GenerationResponse)
async def generate_text(request: GenerationRequest, background_tasks: BackgroundTasks):
    """Generate text (non-streaming)"""
    job_id = await server_instance.submit_generation_request(request)
    
    # For non-streaming, we wait for completion with improved timeout handling
    max_wait_time = 120.0  # Increased to 2 minutes for complex prompts
    start_time = time.time()
    poll_interval = 0.2  # Check more frequently
    
    logger.info(f"Waiting for job {job_id} completion (max {max_wait_time}s)")
    
    while (time.time() - start_time) < max_wait_time:
        try:
            result = await server_instance.get_generation_result(job_id)
            elapsed = time.time() - start_time
            logger.info(f"Job {job_id} completed successfully in {elapsed:.2f}s")
            return result
        except HTTPException as e:
            if e.status_code == 202:  # Still processing
                await asyncio.sleep(poll_interval)
                continue
            elif e.status_code == 404:  # Job not found - may have failed
                logger.warning(f"Job {job_id} not found, may have failed")
                raise HTTPException(status_code=500, detail="Job processing failed")
            else:
                raise
        except Exception as e:
            logger.error(f"Error checking job {job_id}: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    # Timeout occurred
    elapsed = time.time() - start_time
    logger.warning(f"Job {job_id} timed out after {elapsed:.2f}s")
    raise HTTPException(status_code=408, detail=f"Request timeout after {max_wait_time}s")


@app.post("/generate/stream")
async def generate_text_stream(request: GenerationRequest):
    """Generate text with streaming response"""
    return await server_instance.generate_text_streaming(request)


@app.get("/jobs/{job_id}")
async def get_job_status(job_id: str):
    """Get status of a specific job"""
    return server_instance.get_job_status(job_id)


@app.get("/jobs/{job_id}/result", response_model=GenerationResponse)
async def get_job_result(job_id: str):
    """Get result of a completed job"""
    return await server_instance.get_generation_result(job_id)


@app.get("/stats", response_model=SystemStats)
async def get_system_stats():
    """Get system statistics"""
    return server_instance.get_system_stats()


@app.get("/stats/detailed")
async def get_detailed_stats():
    """Get detailed system statistics"""
    return {
        "system": server_instance.get_system_stats().model_dump(),
        "scheduler": server_instance.scheduler.get_queue_stats(),
        "profiler": server_instance.scheduler.get_profiler_statistics(),
        "memory": server_instance.memory_manager.get_memory_stats().__dict__,
        "cache": server_instance.cache_manager.get_cache_stats(),
        "inference": server_instance.inference_engine.get_inference_stats(),
        "uptime_seconds": time.time() - server_instance.start_time
    }


@app.post("/admin/cleanup")
async def cleanup_old_jobs(max_age_hours: float = 24):
    """Clean up old jobs"""
    cleaned_count = await server_instance.cleanup_old_jobs(max_age_hours)
    return {"message": f"Cleaned up {cleaned_count} old jobs"}


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if server_instance.is_running else "unhealthy",
        "model_loaded": server_instance.inference_engine.is_model_loaded,
        "active_jobs": len(server_instance.active_jobs),
        "timestamp": time.time()
    }


@app.get("/")
async def root():
    """Render the main web interface"""
    return HTMLResponse(content=open('templates/index.html').read(), status_code=200)


def run_server(host: str = None, port: int = None, **kwargs):
    """Run the FastServe server"""
    config = get_config()
    host = host or config.host
    port = port or config.port
    
    logger.info(f"Starting FastServe server on {host}:{port}")
    
    # Set default values if not provided
    kwargs.setdefault('reload', False)  # Disable reload for production by default
    kwargs.setdefault('workers', 1)
    
    uvicorn.run(
        "fastserve.server:app",
        host=host,
        port=port,
        **kwargs
    )


if __name__ == "__main__":
    run_server()
