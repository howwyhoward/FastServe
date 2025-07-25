#!/usr/bin/env python3
"""
FastServe Benchmark Script
Tests the performance and effectiveness of FastServe's preemptive scheduling
"""

import asyncio
import httpx
import time
import statistics
import json
from typing import List, Dict, Any
import argparse

class FastServeBenchmark:
    def __init__(self, server_url: str = "http://localhost:8000"):
        self.server_url = server_url
        self.results: List[Dict[str, Any]] = []
    
    async def health_check(self) -> bool:
        """Check if FastServe server is running"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.server_url}/health", timeout=5.0)
                return response.status_code == 200
        except:
            return False
    
    async def single_request_benchmark(self, prompt: str, max_tokens: int = 50) -> Dict[str, Any]:
        """Benchmark a single request"""
        start_time = time.time()
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.server_url}/generate",
                json={
                    "prompt": prompt,
                    "max_new_tokens": max_tokens,
                    "temperature": 0.7
                },
                timeout=120.0
            )
        
        end_time = time.time()
        
        if response.status_code == 200:
            result = response.json()
            return {
                "success": True,
                "wall_time": end_time - start_time,
                "completion_time": result["completion_time"],
                "queue_time": result["queue_time"],
                "execution_time": result["execution_time"],
                "preemption_count": result["preemption_count"],
                "tokens_generated": result["total_tokens"],
                "prompt_length": len(prompt.split()),
                "job_id": result["job_id"]
            }
        else:
            return {
                "success": False,
                "error": f"HTTP {response.status_code}",
                "wall_time": end_time - start_time,
                "prompt_length": len(prompt.split())
            }
    
    async def concurrent_benchmark(self, prompts: List[str], max_tokens: int = 50) -> List[Dict[str, Any]]:
        """Benchmark concurrent requests to test preemptive scheduling"""
        print(f"Running concurrent benchmark with {len(prompts)} requests...")
        
        async def single_request(prompt: str, request_id: int):
            result = await self.single_request_benchmark(prompt, max_tokens)
            result["request_id"] = request_id
            result["prompt"] = prompt[:50] + "..." if len(prompt) > 50 else prompt
            return result
        
        start_time = time.time()
        tasks = [single_request(prompt, i) for i, prompt in enumerate(prompts)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        end_time = time.time()
        
        # Filter out exceptions and add metadata
        valid_results = []
        for result in results:
            if isinstance(result, dict):
                result["total_benchmark_time"] = end_time - start_time
                valid_results.append(result)
        
        return valid_results
    
    async def load_test(self, num_requests: int = 20, max_tokens: int = 30) -> Dict[str, Any]:
        """Run load test with varying prompt lengths"""
        
        # Create prompts of different lengths to test skip-join logic
        short_prompts = [
            "Hello",
            "The weather is",
            "AI will",
            "Today I learned"
        ]
        
        medium_prompts = [
            "The future of artificial intelligence includes many exciting possibilities such as",
            "Climate change is a global challenge that requires immediate action from governments and individuals",
            "Machine learning algorithms have revolutionized many industries by enabling computers to",
            "Space exploration has always fascinated humanity, and recent developments in rocket technology"
        ]
        
        long_prompts = [
            "The history of computer science spans several decades and includes many groundbreaking discoveries that have shaped modern technology. From the early mechanical computers of the 1940s to today's advanced neural networks and quantum computing research, the field has evolved dramatically. Key milestones include the development of programming languages, the invention of the transistor, the creation of the internet, and the recent advances in artificial intelligence and machine learning that are transforming industries worldwide",
            "Renewable energy sources such as solar, wind, and hydroelectric power are becoming increasingly important as the world seeks to reduce its dependence on fossil fuels and combat climate change. These technologies have advanced significantly in recent years, with improved efficiency and reduced costs making them more competitive with traditional energy sources. The transition to renewable energy requires substantial investment in infrastructure, policy changes to support clean energy adoption, and continued research and development to overcome technical challenges",
        ]
        
        # Distribute requests across prompt types
        all_prompts = []
        for _ in range(num_requests // 3):
            all_prompts.extend([
                short_prompts[_ % len(short_prompts)],
                medium_prompts[_ % len(medium_prompts)],
                long_prompts[_ % len(long_prompts)]
            ])
        
        # Fill remaining slots with short prompts
        while len(all_prompts) < num_requests:
            all_prompts.append(short_prompts[len(all_prompts) % len(short_prompts)])
        
        results = await self.concurrent_benchmark(all_prompts, max_tokens)
        
        # Analyze results
        successful_results = [r for r in results if r.get("success", False)]
        
        if not successful_results:
            return {"error": "No successful requests"}
        
        wall_times = [r["wall_time"] for r in successful_results]
        completion_times = [r["completion_time"] for r in successful_results]
        queue_times = [r["queue_time"] for r in successful_results]
        execution_times = [r["execution_time"] for r in successful_results]
        preemption_counts = [r["preemption_count"] for r in successful_results]
        tokens_generated = [r["tokens_generated"] for r in successful_results]
        
        return {
            "total_requests": num_requests,
            "successful_requests": len(successful_results),
            "success_rate": len(successful_results) / num_requests,
            
            "wall_time": {
                "mean": statistics.mean(wall_times),
                "median": statistics.median(wall_times),
                "min": min(wall_times),
                "max": max(wall_times),
                "stdev": statistics.stdev(wall_times) if len(wall_times) > 1 else 0
            },
            
            "completion_time": {
                "mean": statistics.mean(completion_times),
                "median": statistics.median(completion_times),
                "min": min(completion_times),
                "max": max(completion_times),
                "stdev": statistics.stdev(completion_times) if len(completion_times) > 1 else 0
            },
            
            "queue_time": {
                "mean": statistics.mean(queue_times),
                "median": statistics.median(queue_times),
                "total": sum(queue_times)
            },
            
            "execution_time": {
                "mean": statistics.mean(execution_times),
                "median": statistics.median(execution_times),
                "total": sum(execution_times)
            },
            
            "preemption": {
                "total_preemptions": sum(preemption_counts),
                "mean_per_job": statistics.mean(preemption_counts),
                "max_preemptions": max(preemption_counts),
                "jobs_with_preemptions": len([p for p in preemption_counts if p > 0])
            },
            
            "throughput": {
                "total_tokens": sum(tokens_generated),
                "tokens_per_second": sum(tokens_generated) / sum(execution_times) if sum(execution_times) > 0 else 0,
                "requests_per_second": len(successful_results) / max(wall_times) if wall_times else 0
            },
            
            "individual_results": successful_results[:10]  # Show first 10 for analysis
        }
    
    async def system_stats_analysis(self) -> Dict[str, Any]:
        """Get detailed system statistics"""
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{self.server_url}/stats/detailed")
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"Failed to get stats: {response.status_code}"}
    
    async def run_full_benchmark(self, num_requests: int = 20) -> Dict[str, Any]:
        """Run comprehensive benchmark suite"""
        print("FastServe Performance Benchmark")
        print("=" * 50)
        
        # Health check
        if not await self.health_check():
            return {"error": "FastServe server is not running or not healthy"}
        
        print("✓ Server health check passed")
        
        # Get initial system stats
        initial_stats = await self.system_stats_analysis()
        print("✓ Initial system stats collected")
        
        # Run load test
        print(f"Running load test with {num_requests} requests...")
        load_test_results = await self.load_test(num_requests)
        
        # Get final system stats
        final_stats = await self.system_stats_analysis()
        
        return {
            "benchmark_timestamp": time.time(),
            "load_test": load_test_results,
            "initial_system_stats": initial_stats,
            "final_system_stats": final_stats
        }
    
    def print_results(self, results: Dict[str, Any]):
        """Print benchmark results in a readable format"""
        if "error" in results:
            print(f"❌ Benchmark failed: {results['error']}")
            return
        
        load_test = results["load_test"]
        
        print("\n📊 LOAD TEST RESULTS")
        print("-" * 30)
        print(f"Total Requests: {load_test['total_requests']}")
        print(f"Successful: {load_test['successful_requests']}")
        print(f"Success Rate: {load_test['success_rate']:.1%}")
        
        print("\n⏱️  TIMING METRICS")
        print("-" * 20)
        wt = load_test["wall_time"]
        print(f"Wall Time - Mean: {wt['mean']:.2f}s, Median: {wt['median']:.2f}s, Max: {wt['max']:.2f}s")
        
        ct = load_test["completion_time"]
        print(f"Completion Time - Mean: {ct['mean']:.2f}s, Median: {ct['median']:.2f}s")
        
        qt = load_test["queue_time"]
        print(f"Queue Time - Mean: {qt['mean']:.3f}s, Total: {qt['total']:.2f}s")
        
        print("\n🔄 PREEMPTION ANALYSIS")
        print("-" * 25)
        p = load_test["preemption"]
        print(f"Total Preemptions: {p['total_preemptions']}")
        print(f"Mean per Job: {p['mean_per_job']:.1f}")
        print(f"Jobs with Preemptions: {p['jobs_with_preemptions']}/{load_test['successful_requests']}")
        print(f"Max Preemptions (single job): {p['max_preemptions']}")
        
        print("\n🚀 THROUGHPUT METRICS")
        print("-" * 22)
        tp = load_test["throughput"]
        print(f"Total Tokens Generated: {tp['total_tokens']}")
        print(f"Tokens per Second: {tp['tokens_per_second']:.1f}")
        print(f"Requests per Second: {tp['requests_per_second']:.1f}")
        
        # Show system stats if available
        if "final_system_stats" in results and "error" not in results["final_system_stats"]:
            final_stats = results["final_system_stats"]
            print("\n📈 SYSTEM STATISTICS")
            print("-" * 21)
            print(f"Memory - GPU: {final_stats['memory']['gpu_utilization']:.1f}%, CPU: {final_stats['memory']['cpu_utilization']:.1f}%")
            print(f"Cache Hit Rate: {final_stats['cache']['hit_rate_percent']:.1f}%")
            print(f"Uptime: {final_stats['uptime_seconds']:.1f}s")

async def main():
    parser = argparse.ArgumentParser(description="FastServe Benchmark Suite")
    parser.add_argument("--server", default="http://localhost:8000", help="FastServe server URL")
    parser.add_argument("--requests", type=int, default=20, help="Number of requests for load test")
    parser.add_argument("--output", help="Output file for results (JSON)")
    
    args = parser.parse_args()
    
    benchmark = FastServeBenchmark(args.server)
    results = await benchmark.run_full_benchmark(args.requests)
    
    benchmark.print_results(results)
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Detailed results saved to {args.output}")

if __name__ == "__main__":
    asyncio.run(main())
