"""
Inference Engine for FastServe
Implements preemptive text generation with token-level scheduling
"""

import asyncio
import logging
import time
from typing import Optional, AsyncIterator, Dict, Any, List
import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    GenerationConfig, StoppingCriteria, StoppingCriteriaList
)

from .models import InferenceJob, JobStatus
from .scheduler import SkipJoinMLFQScheduler
from .cache_manager import KeyValueCacheManager
from .memory_manager import MemoryManager
from .config import get_config
from .ollama_adapter import OllamaAdapter

logger = logging.getLogger(__name__)


class PreemptionStoppingCriteria(StoppingCriteria):
    """Custom stopping criteria for preemptive generation"""
    
    def __init__(self, scheduler: SkipJoinMLFQScheduler, max_tokens_per_step: int = 1):
        self.scheduler = scheduler
        self.max_tokens_per_step = max_tokens_per_step
        self.tokens_generated_this_step = 0
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        self.tokens_generated_this_step += 1
        
        # Check if we should preempt after generating one token
        if self.tokens_generated_this_step >= self.max_tokens_per_step:
            should_preempt = self.scheduler.should_preempt_current_job()
            if should_preempt:
                logger.debug("Preemption requested, stopping generation")
                return True
        
        return False


class InferenceEngine:
    """
    Preemptive Inference Engine for FastServe
    
    Key features:
    - Token-level preemption using custom stopping criteria
    - Integration with scheduler and cache manager
    - Support for incremental generation with cache reuse
    - Memory-efficient model loading for Mac M2
    """
    
    def __init__(self, scheduler: SkipJoinMLFQScheduler, 
                 cache_manager: KeyValueCacheManager, config=None):
        self.config = config or get_config()
        self.scheduler = scheduler
        self.cache_manager = cache_manager
        self.device = self.config.device
        
        # Backend selection
        self.use_ollama = self.config.use_ollama
        
        if self.use_ollama:
            # Initialize Ollama adapter
            self.ollama_adapter = OllamaAdapter(self.config)
            self.model = None
            self.tokenizer = None
            self.generation_config = None
            logger.info(f"Inference Engine initialized with Ollama backend: {self.config.ollama_model}")
        else:
            # Initialize HuggingFace backend
            self.ollama_adapter = None
            self.model: Optional[AutoModelForCausalLM] = None
            self.tokenizer: Optional[AutoTokenizer] = None
            self.generation_config: Optional[GenerationConfig] = None
            logger.info(f"Inference Engine initialized with HuggingFace backend for device: {self.device}")
        
        # Generation state
        self.is_model_loaded = False
        self.current_generation_task: Optional[asyncio.Task] = None
        
        # Statistics
        self.total_tokens_generated = 0
        self.total_jobs_processed = 0
        self.total_preemptions = 0
        
        # Job completion callback
        self.job_completion_callback = None
    
    async def load_model(self, model_name: str = None) -> bool:
        """
        Load model and tokenizer with Mac M2 optimizations
        """
        model_name = model_name or self.config.model_name
        
        try:
            logger.info(f"Loading model: {model_name}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Set up pad token properly for DialoGPT and similar models
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                logger.info(f"Set pad_token to eos_token for model {model_name}")
            
            # Ensure we have proper token IDs
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
                logger.info(f"Set pad_token_id to eos_token_id for model {model_name}")
            
            logger.info(f"Tokenizer config: eos_token_id={self.tokenizer.eos_token_id}, pad_token_id={self.tokenizer.pad_token_id}")
            
            # Load model with Mac M2 optimizations
            model_kwargs = {
                "trust_remote_code": True,
                "torch_dtype": self.config.dtype,
                "low_cpu_mem_usage": True,
            }
            
            # Only specify device_map for MPS/GPU
            if self.device != "cpu":
                model_kwargs["device_map"] = "auto"
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                **model_kwargs
            )
            
            # Move to device if not already there
            if self.device == "mps":
                self.model = self.model.to("mps")
            elif self.device == "cpu":
                self.model = self.model.to("cpu")
            
            # Set model to evaluation mode
            self.model.eval()
            
            # Configure generation
            self.generation_config = GenerationConfig(
                max_new_tokens=self.config.max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                top_k=50,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
                return_dict_in_generate=True,
                output_scores=True,
                repetition_penalty=1.1,  # Avoid repetitive generation
                no_repeat_ngram_size=2,  # Avoid repeating n-grams
                early_stopping=True,  # Stop when eos_token is generated
                min_new_tokens=1,  # Ensure at least one token is generated
                max_time=30.0  # Maximum generation time in seconds
            )
            
            self.is_model_loaded = True
            logger.info(f"Model {model_name} loaded successfully on {self.device}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            return False
    
    async def generate_text(self, job: InferenceJob) -> bool:
        """
        Generate text for a job with preemptive scheduling
        """
        if not self.is_model_loaded:
            logger.error("Model not loaded")
            return False
        
        # Route to appropriate backend
        if self.use_ollama:
            return await self._generate_text_ollama(job)
        else:
            return await self._generate_text_huggingface(job)
    
    async def _generate_text_ollama(self, job: InferenceJob) -> bool:
        """Generate text using Ollama backend"""
        try:
            success = await self.ollama_adapter.generate_text_sync(job)
            if success:
                self.total_tokens_generated += job.total_tokens_generated
                self.total_jobs_processed += 1
                logger.info(f"Job {job.job_id} completed via Ollama: {job.total_tokens_generated} tokens generated")
            return success
        except Exception as e:
            logger.error(f"Error generating text via Ollama for job {job.job_id}: {e}")
            job.update_status(JobStatus.FAILED)
            return False
    
    async def _generate_text_huggingface(self, job: InferenceJob) -> bool:
        """Generate text using HuggingFace backend"""
        try:
            # Tokenize input
            inputs = self.tokenizer(
                job.prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config.max_sequence_length
            )
            
            # Move inputs to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get cached key-value states if available
            past_key_values = await self.cache_manager.get_cache_for_generation(
                job, self.model.config.num_hidden_layers
            )
            
            # Update job tracking
            job.current_position = inputs['input_ids'].size(-1)
            if past_key_values:
                # Adjust position for cached tokens
                job.current_position += past_key_values[0][0].size(-2)
            
            # Create preemption stopping criteria
            preemption_criteria = PreemptionStoppingCriteria(
                self.scheduler, 
                max_tokens_per_step=1  # Generate one token at a time
            )
            
            stopping_criteria = StoppingCriteriaList([preemption_criteria])
            
            # Generate tokens incrementally
            generated_tokens = []
            generation_complete = False
            consecutive_empty_generations = 0
            max_empty_generations = 3  # Fail after 3 consecutive empty generations
            
            while not generation_complete and len(generated_tokens) < job.max_new_tokens:
                # Reset preemption criteria for each step
                preemption_criteria.tokens_generated_this_step = 0
                
                # Generate next token(s)
                with torch.no_grad():
                    try:
                        outputs = self.model.generate(
                            input_ids=inputs['input_ids'],
                            attention_mask=inputs.get('attention_mask'),
                            max_new_tokens=min(3, job.max_new_tokens - len(generated_tokens)),  # Even smaller batches
                            temperature=job.temperature,
                            top_p=job.top_p,
                            top_k=job.top_k,
                            do_sample=job.do_sample,
                            pad_token_id=self.tokenizer.pad_token_id,
                            eos_token_id=self.tokenizer.eos_token_id,
                            use_cache=True,
                            return_dict_in_generate=True
                        )
                    except Exception as e:
                        logger.error(f"Model.generate() failed for job {job.job_id}: {e}")
                        logger.error(f"Input shape: {inputs['input_ids'].shape}, Device: {inputs['input_ids'].device}")
                        raise e
                
                # Extract new tokens with bounds checking
                original_length = inputs['input_ids'].size(-1)
                
                # Debug logging
                logger.debug(f"Job {job.job_id}: Original input length: {original_length}")
                logger.debug(f"Job {job.job_id}: Output sequences count: {len(outputs.sequences)}")
                if len(outputs.sequences) > 0:
                    logger.debug(f"Job {job.job_id}: First sequence shape: {outputs.sequences[0].shape}")
                
                # Check if outputs.sequences is empty or malformed
                if len(outputs.sequences) == 0:
                    logger.warning(f"Empty sequences returned for job {job.job_id}, stopping generation")
                    break
                
                generated_sequence = outputs.sequences[0]
                
                # Additional safety check
                if generated_sequence.numel() == 0:
                    logger.warning(f"Zero-element sequence returned for job {job.job_id}, stopping generation")
                    break
                
                if generated_sequence.size(-1) > original_length:
                    new_tokens = generated_sequence[original_length:]
                    generated_tokens.extend(new_tokens.tolist())
                    consecutive_empty_generations = 0  # Reset counter
                else:
                    # No new tokens generated, possibly due to early stopping
                    new_tokens = torch.tensor([], dtype=torch.long, device=self.device)
                    consecutive_empty_generations += 1
                    logger.debug(f"No new tokens generated for job {job.job_id} (attempt {consecutive_empty_generations})")
                
                # If no tokens were generated multiple times, break to avoid infinite loop
                if consecutive_empty_generations >= max_empty_generations:
                    logger.warning(f"Generation failed for job {job.job_id} - no tokens generated after {max_empty_generations} attempts")
                    break
                
                # If no tokens were generated this round, continue to next iteration
                if len(new_tokens) == 0:
                    continue
                
                # Update cache
                if hasattr(outputs, 'past_key_values') and outputs.past_key_values:
                    await self.cache_manager.update_cache_from_generation(job, outputs.past_key_values)
                    past_key_values = outputs.past_key_values
                
                # Update job state
                job.generated_tokens = generated_tokens
                job.total_tokens_generated = len(generated_tokens)
                job.generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                
                # Check for completion - only check if we have new tokens
                if len(new_tokens) > 0 and self.tokenizer.eos_token_id in new_tokens:
                    generation_complete = True
                    logger.debug(f"Generation completed for job {job.job_id}")
                
                # Check for preemption
                if self.scheduler.should_preempt_current_job():
                    logger.debug(f"Job {job.job_id} preempted after {len(generated_tokens)} tokens")
                    return False  # Preempted, will resume later
                
                # Update inputs for next iteration
                inputs['input_ids'] = outputs.sequences
                if 'attention_mask' in inputs and len(new_tokens) > 0:
                    # Extend attention mask only if new tokens were generated
                    new_attention = torch.ones(
                        (inputs['attention_mask'].size(0), new_tokens.size(-1)),
                        dtype=inputs['attention_mask'].dtype,
                        device=inputs['attention_mask'].device
                    )
                    inputs['attention_mask'] = torch.cat([inputs['attention_mask'], new_attention], dim=-1)
            
            # Job completed successfully
            self.total_tokens_generated += len(generated_tokens)
            logger.info(f"Job {job.job_id} completed: {len(generated_tokens)} tokens generated")
            return True
            
        except Exception as e:
            logger.error(f"Error generating text for job {job.job_id}: {e}")
            job.update_status(JobStatus.FAILED)
            # Increment retry count to track failures
            job.retry_count += 1
            return False
    
    async def run_inference_loop(self) -> None:
        """
        Main inference loop with preemptive scheduling
        """
        logger.info("Starting inference loop")
        
        while True:
            try:
                # Get next job from scheduler
                job = await self.scheduler.get_next_job()
                
                if job is None:
                    # No jobs available, wait briefly
                    await asyncio.sleep(0.1)
                    continue
                
                logger.debug(f"Processing job {job.job_id}")
                
                # Generate text for the job
                completed = await self.generate_text(job)
                
                if completed:
                    # Job completed
                    job.update_status(JobStatus.COMPLETED)
                    self.scheduler.complete_current_job()
                    self.total_jobs_processed += 1
                    
                    # Clean up cache for completed job
                    await self.cache_manager.invalidate_job_cache(job.job_id)
                    
                    # Notify server of completion
                    if self.job_completion_callback:
                        self.job_completion_callback(job, True)
                else:
                    # Job was preempted
                    self.total_preemptions += 1
                    await self.scheduler.preempt_current_job()
                
            except Exception as e:
                logger.error(f"Error in inference loop: {e}")
                
                # If there was a current job, mark it as failed
                current_job = getattr(self.scheduler, 'current_job', None)
                if current_job:
                    current_job.update_status(JobStatus.FAILED)
                    if self.job_completion_callback:
                        self.job_completion_callback(current_job, False)
                    self.scheduler.current_job = None
                
                # Sleep longer to prevent tight error loops
                await asyncio.sleep(2.0)
    
    async def generate_streaming(self, job: InferenceJob) -> AsyncIterator[str]:
        """
        Generate text with streaming output
        """
        if not self.is_model_loaded:
            raise RuntimeError("Model not loaded")
        
        # Similar to generate_text but yields tokens as they're generated
        inputs = self.tokenizer(
            job.prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_sequence_length
        )
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        past_key_values = await self.cache_manager.get_cache_for_generation(
            job, self.model.config.num_hidden_layers
        )
        
        generated_tokens = []
        
        while len(generated_tokens) < job.max_new_tokens:
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs.get('attention_mask'),
                    past_key_values=past_key_values,
                    max_new_tokens=1,  # One token at a time for streaming
                    temperature=job.temperature,
                    top_p=job.top_p,
                    top_k=job.top_k,
                    do_sample=job.do_sample,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=True,
                    return_dict_in_generate=True
                )
            
            # Check if outputs.sequences is empty or malformed
            if len(outputs.sequences) == 0:
                logger.warning(f"Empty sequences returned for streaming job {job.job_id}, stopping")
                break
            
            sequence = outputs.sequences[0]
            if sequence.size(-1) == 0:
                logger.warning(f"Empty sequence returned for streaming job {job.job_id}, stopping")
                break
                
            # Additional safety check for the original input length
            original_length = inputs['input_ids'].size(-1)
            if sequence.size(-1) <= original_length:
                logger.warning(f"No new tokens generated for streaming job {job.job_id}, stopping")
                break
                
            # Get new token safely
            new_token = sequence[-1].item()
            generated_tokens.append(new_token)
            
            # Decode and yield token
            token_text = self.tokenizer.decode([new_token], skip_special_tokens=True)
            yield token_text
            
            # Update cache
            if hasattr(outputs, 'past_key_values') and outputs.past_key_values:
                await self.cache_manager.update_cache_from_generation(job, outputs.past_key_values)
                past_key_values = outputs.past_key_values
            
            # Check for completion
            if new_token == self.tokenizer.eos_token_id:
                break
            
            # Update inputs for next iteration
            inputs['input_ids'] = outputs.sequences
    
    def get_inference_stats(self) -> Dict[str, Any]:
        """Get inference engine statistics"""
        if self.use_ollama and self.ollama_adapter:
            # Get Ollama-specific stats
            ollama_stats = self.ollama_adapter.get_inference_stats()
            # Combine with engine stats
            ollama_stats.update({
                'total_preemptions': self.total_preemptions,
                'preemption_rate': (
                    self.total_preemptions / max(1, ollama_stats['total_requests_processed'])
                )
            })
            return ollama_stats
        else:
            # Return HuggingFace stats
            return {
                'model_loaded': self.is_model_loaded,
                'model_name': self.config.model_name,
                'device': self.device,
                'backend': 'HuggingFace',
                'total_tokens_generated': self.total_tokens_generated,
                'total_jobs_processed': self.total_jobs_processed,
                'total_preemptions': self.total_preemptions,
                'avg_tokens_per_job': (
                    self.total_tokens_generated / max(1, self.total_jobs_processed)
                ),
                'preemption_rate': (
                    self.total_preemptions / max(1, self.total_jobs_processed)
                ),
                'supports_preemption': True,
                'supports_streaming': True
            }
    
    async def start(self) -> None:
        """Start inference engine"""
        if self.use_ollama:
            await self.ollama_adapter.start()
            self.is_model_loaded = self.ollama_adapter.is_model_loaded
        else:
            if not self.is_model_loaded:
                await self.load_model()
        
        logger.info("Inference Engine started")
    
    async def stop(self) -> None:
        """Stop inference engine"""
        if self.current_generation_task:
            self.current_generation_task.cancel()
        
        if self.use_ollama and self.ollama_adapter:
            await self.ollama_adapter.stop()
        
        logger.info("Inference Engine stopped")
