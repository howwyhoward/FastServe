"""
Ollama Adapter for FastServe
Integrates local Ollama models with FastServe's inference system
"""

import asyncio
import json
import logging
import subprocess
import time
from typing import AsyncGenerator, Optional, Dict, Any, List
import aiohttp
import requests

from .models import InferenceJob
from .config import get_config

logger = logging.getLogger(__name__)


class OllamaAdapter:
    """
    Adapter to use local Ollama models with FastServe
    
    Provides the same interface as the HuggingFace inference engine
    but uses Ollama's local API for generation
    """
    
    def __init__(self, config=None):
        self.config = config or get_config()
        
        # Ollama settings
        self.ollama_host = "http://localhost:11434"
        self.model_name = "deepseek-r1:14b"  # Default to DeepSeek
        self.is_model_loaded = False
        self.generation_session = None
        
        # Statistics
        self.total_tokens_generated = 0
        self.total_requests_processed = 0
        self.average_tokens_per_second = 0.0
        
        logger.info(f"Ollama Adapter initialized for model: {self.model_name}")
    
    async def start(self) -> None:
        """Start the Ollama adapter"""
        try:
            # Check if Ollama is running
            await self._ensure_ollama_running()
            
            # Verify model is available
            await self._verify_model_available()
            
            self.is_model_loaded = True
            logger.info(f"Ollama adapter started successfully with {self.model_name}")
            
        except Exception as e:
            logger.error(f"Failed to start Ollama adapter: {e}")
            raise
    
    async def stop(self) -> None:
        """Stop the Ollama adapter"""
        self.is_model_loaded = False
        if self.generation_session:
            await self.generation_session.close()
        logger.info("Ollama adapter stopped")
    
    async def _ensure_ollama_running(self) -> bool:
        """Ensure Ollama service is running"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.ollama_host}/api/tags", timeout=5) as response:
                    if response.status == 200:
                        logger.info("Ollama service is running")
                        return True
        except Exception as e:
            logger.error(f"Ollama service not available: {e}")
            raise RuntimeError("Ollama service is not running. Please start with 'ollama serve'")
    
    async def _verify_model_available(self) -> bool:
        """Verify the specified model is available locally"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.ollama_host}/api/tags") as response:
                    if response.status == 200:
                        data = await response.json()
                        available_models = [model['name'] for model in data.get('models', [])]
                        
                        if self.model_name in available_models:
                            logger.info(f"Model {self.model_name} is available")
                            return True
                        else:
                            logger.error(f"Model {self.model_name} not found. Available: {available_models}")
                            raise RuntimeError(f"Model {self.model_name} not found locally")
        except Exception as e:
            logger.error(f"Failed to verify model: {e}")
            raise
    
    async def generate_text_sync(self, job: InferenceJob) -> bool:
        """
        Generate text synchronously (non-streaming)
        Compatible with existing FastServe interface
        """
        try:
            start_time = time.time()
            
            # Prepare the generation request
            payload = {
                "model": self.model_name,
                "prompt": job.prompt,
                "stream": False,
                "options": {
                    "temperature": job.temperature,
                    "top_p": job.top_p,
                    "top_k": job.top_k,
                    "num_predict": job.max_new_tokens,
                }
            }
            
            # Make the request
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.ollama_host}/api/generate",
                    json=payload,
                    timeout=120  # 2 minute timeout
                ) as response:
                    
                    if response.status == 200:
                        result = await response.json()
                        
                        # Update job with results
                        job.generated_text = result.get('response', '')
                        job.total_tokens_generated = self._count_tokens(job.generated_text)
                        job.update_status(job.status)  # Keep current status
                        
                        # Update statistics
                        generation_time = time.time() - start_time
                        self.total_tokens_generated += job.total_tokens_generated
                        self.total_requests_processed += 1
                        self.average_tokens_per_second = job.total_tokens_generated / max(generation_time, 0.1)
                        
                        logger.info(f"Generated {job.total_tokens_generated} tokens in {generation_time:.2f}s "
                                  f"({self.average_tokens_per_second:.1f} tokens/sec)")
                        
                        return True
                    else:
                        error_text = await response.text()
                        logger.error(f"Ollama generation failed: {response.status} - {error_text}")
                        return False
                        
        except Exception as e:
            logger.error(f"Error during text generation: {e}")
            return False
    
    async def generate_text_streaming(self, job: InferenceJob) -> AsyncGenerator[str, None]:
        """
        Generate text with streaming (word-by-word)
        For real-time applications
        """
        try:
            payload = {
                "model": self.model_name,
                "prompt": job.prompt,
                "stream": True,
                "options": {
                    "temperature": job.temperature,
                    "top_p": job.top_p,
                    "top_k": job.top_k,
                    "num_predict": job.max_new_tokens,
                }
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.ollama_host}/api/generate",
                    json=payload,
                    timeout=300  # 5 minute timeout for streaming
                ) as response:
                    
                    if response.status == 200:
                        full_response = ""
                        
                        async for line in response.content:
                            if line:
                                try:
                                    data = json.loads(line.decode('utf-8'))
                                    
                                    if 'response' in data:
                                        token = data['response']
                                        full_response += token
                                        job.generated_text = full_response
                                        job.total_tokens_generated = self._count_tokens(full_response)
                                        
                                        yield token
                                        
                                    if data.get('done', False):
                                        break
                                        
                                except json.JSONDecodeError:
                                    continue
                    else:
                        logger.error(f"Streaming failed: {response.status}")
                        yield f"Error: Failed to start streaming generation"
                        
        except Exception as e:
            logger.error(f"Streaming generation error: {e}")
            yield f"Error: {str(e)}"
    
    def _count_tokens(self, text: str) -> int:
        """
        Simple token counting (approximation)
        In a real implementation, you'd use the actual tokenizer
        """
        # Rough approximation: 1 token ≈ 4 characters for most models
        return max(1, len(text) // 4)
    
    async def preempt_generation(self) -> bool:
        """
        Preempt current generation
        Note: Ollama doesn't support mid-generation preemption like HuggingFace
        This is a limitation of the Ollama API
        """
        logger.warning("Preemption not supported with Ollama API")
        return False
    
    def get_inference_stats(self) -> Dict[str, Any]:
        """Get inference engine statistics"""
        return {
            "model_name": self.model_name,
            "model_loaded": self.is_model_loaded,
            "total_tokens_generated": self.total_tokens_generated,
            "total_requests_processed": self.total_requests_processed,
            "average_tokens_per_second": self.average_tokens_per_second,
            "backend": "Ollama",
            "supports_preemption": False,
            "supports_streaming": True
        }
    
    async def get_available_models(self) -> List[str]:
        """Get list of available Ollama models"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.ollama_host}/api/tags") as response:
                    if response.status == 200:
                        data = await response.json()
                        return [model['name'] for model in data.get('models', [])]
        except Exception as e:
            logger.error(f"Failed to get available models: {e}")
        return []
    
    async def switch_model(self, model_name: str) -> bool:
        """Switch to a different Ollama model"""
        try:
            # Check if model is available
            available_models = await self.get_available_models()
            if model_name not in available_models:
                logger.error(f"Model {model_name} not available. Available: {available_models}")
                return False
            
            self.model_name = model_name
            logger.info(f"Switched to model: {model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to switch model: {e}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about current model"""
        return {
            "name": self.model_name,
            "backend": "Ollama",
            "local": True,
            "context_length": 131072 if "deepseek" in self.model_name.lower() else "unknown",
            "parameters": "14.8B" if "deepseek-r1:14b" in self.model_name else "unknown"
        } 