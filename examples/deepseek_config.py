#!/usr/bin/env python3
"""
Example: How to configure FastServe to use DeepSeek R1 14B model
Shows how to enable Ollama backend instead of HuggingFace
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastserve import FastServeServer
from fastserve.config import FastServeConfig
from fastserve.models import GenerationRequest

async def main():
    """Example using DeepSeek with FastServe"""
    
    print("🚀 FastServe + DeepSeek R1 14B Example")
    print("=" * 50)
    
    # Create configuration for DeepSeek
    config = FastServeConfig(
        # Enable Ollama backend
        use_ollama=True,
        ollama_model="deepseek-r1:14b",
        
        # Generation settings
        max_new_tokens=200,
        
        # Scheduler settings
        num_priority_queues=4,
        queue_capacity=50,
        
        # Host/port for server
        host="127.0.0.1",
        port=8000
    )
    
    # Initialize server with DeepSeek config
    server = FastServeServer(config)
    
    try:
        # Start the server
        print("🏗️ Starting FastServe with DeepSeek...")
        await server.start_all_components()
        print("✅ Server started successfully!")
        
        # Test generation
        test_prompts = [
            "What is machine learning?",
            "Write a Python function to sort a list",
            "Explain the theory of relativity in simple terms"
        ]
        
        for i, prompt in enumerate(test_prompts, 1):
            print(f"\n📝 Test {i}: {prompt}")
            
            # Create generation request
            request = GenerationRequest(
                prompt=prompt,
                max_new_tokens=150,
                temperature=0.7
            )
            
            # Submit and get result
            job_id = await server.submit_generation_request(request)
            result = await server.get_generation_result(job_id)
            
            print(f"✅ Generated {result.total_tokens} tokens")
            print(f"📄 Response: {result.generated_text[:100]}...")
        
        # Show statistics
        stats = server.get_system_stats()
        print(f"\n📊 System Statistics:")
        print(f"   Active jobs: {stats.active_jobs}")
        print(f"   Completed jobs: {stats.completed_jobs}")
        print(f"   Avg execution time: {stats.average_execution_time:.2f}s")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    
    finally:
        # Clean shutdown
        await server.stop_all_components()
        print("\n✅ Server stopped")

if __name__ == "__main__":
    print("Prerequisites:")
    print("1. Ollama running: ollama serve")  
    print("2. DeepSeek model: ollama pull deepseek-r1:14b")
    print("3. Dependencies: pip install aiohttp")
    print()
    
    asyncio.run(main()) 