#!/usr/bin/env python3
"""
Example client for FastServe
Demonstrates how to use the FastServe API
"""

import asyncio
import httpx
import json
import time

# Server configuration
SERVER_URL = "http://localhost:8000"

async def simple_generation_example():
    """Example of simple text generation"""
    print("=== Simple Generation Example ===")
    
    async with httpx.AsyncClient() as client:
        # Generate text
        response = await client.post(
            f"{SERVER_URL}/generate",
            json={
                "prompt": "The future of artificial intelligence is",
                "max_new_tokens": 50,
                "temperature": 0.7,
                "top_p": 0.9,
                "do_sample": True
            },
            timeout=60.0
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"Job ID: {result['job_id']}")
            print(f"Generated text: {result['generated_text']}")
            print(f"Total tokens: {result['total_tokens']}")
            print(f"Completion time: {result['completion_time']:.2f}s")
            print(f"Preemptions: {result['preemption_count']}")
        else:
            print(f"Error: {response.status_code} - {response.text}")

async def streaming_generation_example():
    """Example of streaming text generation"""
    print("\n=== Streaming Generation Example ===")
    
    async with httpx.AsyncClient() as client:
        async with client.stream(
            "POST",
            f"{SERVER_URL}/generate/stream",
            json={
                "prompt": "Once upon a time in a land far away,",
                "max_new_tokens": 100,
                "temperature": 0.8,
                "stream": True
            },
            timeout=60.0
        ) as response:
            if response.status_code == 200:
                print("Streaming response:")
                async for chunk in response.aiter_text():
                    if chunk.startswith("data: "):
                        data = json.loads(chunk[6:])
                        if not data.get("is_final", False):
                            print(data["token"], end="", flush=True)
                        else:
                            print(f"\n\nStream completed. Total tokens: {data['total_tokens']}")
            else:
                print(f"Error: {response.status_code} - {await response.atext()}")

async def system_stats_example():
    """Example of getting system statistics"""
    print("\n=== System Stats Example ===")
    
    async with httpx.AsyncClient() as client:
        # Get basic stats
        response = await client.get(f"{SERVER_URL}/stats")
        if response.status_code == 200:
            stats = response.json()
            print("System Statistics:")
            for key, value in stats.items():
                print(f"  {key}: {value}")
        
        # Get detailed stats
        print("\nDetailed Statistics:")
        response = await client.get(f"{SERVER_URL}/stats/detailed")
        if response.status_code == 200:
            detailed_stats = response.json()
            print(f"Uptime: {detailed_stats['uptime_seconds']:.1f} seconds")
            print(f"Scheduler queues: {len(detailed_stats['scheduler'])}")
            print(f"Memory usage: GPU {detailed_stats['memory']['gpu_utilization']:.1f}%, CPU {detailed_stats['memory']['cpu_utilization']:.1f}%")
            print(f"Cache hit rate: {detailed_stats['cache']['hit_rate_percent']:.1f}%")

async def load_test_example():
    """Example of load testing with multiple concurrent requests"""
    print("\n=== Load Test Example ===")
    
    prompts = [
        "The benefits of renewable energy include",
        "Machine learning algorithms are designed to",
        "Climate change affects our planet by",
        "The history of space exploration shows",
        "Modern technology has revolutionized"
    ]
    
    async def send_request(client, prompt, request_id):
        start_time = time.time()
        try:
            response = await client.post(
                f"{SERVER_URL}/generate",
                json={
                    "prompt": prompt,
                    "max_new_tokens": 30,
                    "temperature": 0.7
                },
                timeout=60.0
            )
            end_time = time.time()
            
            if response.status_code == 200:
                result = response.json()
                print(f"Request {request_id}: SUCCESS ({end_time - start_time:.2f}s, {result['preemption_count']} preemptions)")
                return True
            else:
                print(f"Request {request_id}: FAILED - {response.status_code}")
                return False
        except Exception as e:
            print(f"Request {request_id}: ERROR - {e}")
            return False
    
    async with httpx.AsyncClient() as client:
        print(f"Sending {len(prompts)} concurrent requests...")
        start_time = time.time()
        
        tasks = [
            send_request(client, prompt, i) 
            for i, prompt in enumerate(prompts)
        ]
        
        results = await asyncio.gather(*tasks)
        end_time = time.time()
        
        success_count = sum(results)
        total_time = end_time - start_time
        
        print(f"\nLoad test completed:")
        print(f"  Successful requests: {success_count}/{len(prompts)}")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Average time per request: {total_time/len(prompts):.2f}s")

async def health_check_example():
    """Example of health check"""
    print("\n=== Health Check Example ===")
    
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{SERVER_URL}/health")
        if response.status_code == 200:
            health = response.json()
            print(f"Server status: {health['status']}")
            print(f"Model loaded: {health['model_loaded']}")
            print(f"Active jobs: {health['active_jobs']}")
        else:
            print(f"Health check failed: {response.status_code}")

async def main():
    """Run all examples"""
    print("FastServe Client Examples")
    print("=" * 50)
    
    try:
        # Check if server is running
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{SERVER_URL}/health", timeout=5.0)
            if response.status_code != 200:
                print("Server is not running or not healthy!")
                return
        
        # Run examples
        await health_check_example()
        await simple_generation_example()
        await streaming_generation_example()
        await system_stats_example()
        await load_test_example()
        
        print("\n=== All examples completed! ===")
        
    except httpx.ConnectError:
        print(f"Cannot connect to server at {SERVER_URL}")
        print("Make sure FastServe is running with: python main.py")
    except Exception as e:
        print(f"Error running examples: {e}")

if __name__ == "__main__":
    asyncio.run(main())
