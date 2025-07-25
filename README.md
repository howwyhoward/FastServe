# FastServe

**Fast AI Text Generation Server for Mac M2**

FastServe is a high-performance AI server with smart scheduling that makes text generation faster and more efficient on Mac M2.

## ⚡ Super Quick Start

```bash
# 1. Install (one time)
./install.sh

# 2. Start server
source fastserve_env/bin/activate
python main.py

# 3. Test it works
curl http://localhost:8000/health
```

📖 **Want more details?** See [DOCUMENTATION.md](DOCUMENTATION.md) for complete setup and usage guide

---

## 🔬 Technical Details

FastServe implements research from "Fast Distributed Inference Serving for Large Language Models" with preemptive scheduling and skip-join Multi-Level Feedback Queue (MLFQ) algorithms to minimize job completion times and improve system throughput.

## 🎯 Key Features

- **🧠 Intelligent Job Profiler**: Predicts job characteristics and complexity for optimal initial queue assignment
- **⚡ Preemptive Scheduling**: Token-level preemption allows higher-priority jobs to interrupt longer-running inference tasks
- **🔄 Skip-Join MLFQ**: Intelligent queue management that assigns jobs based on profiling to reduce head-of-line blocking
- **🤖 Dual Backend Support**: Use HuggingFace models OR local Ollama models (DeepSeek R1 14B support!)
- **💾 Memory Management**: Efficient GPU/CPU cache swapping with LRU eviction policies
- **🗄️ Key-Value Caching**: Optimized caching for Transformer models with incremental updates
- **🖥️ Mac M2 Optimized**: Native support for Apple Silicon MPS acceleration
- **📡 RESTful API**: FastAPI-based server with streaming and batch inference endpoints
- **📊 Real-time Monitoring**: Comprehensive statistics and performance metrics

## 🏗️ Architecture

FastServe implements the research from "Fast Distributed Inference Serving for Large Language Models" with the following core components:

### Core Components

1. **Job Profiler**: Analyzes incoming requests to predict execution characteristics and determine optimal queue placement
2. **Skip-Join MLFQ Scheduler**: Manages job queues with intelligent initial assignment and preemptive scheduling
3. **Memory Manager**: Handles GPU/CPU memory allocation and cache swapping  
4. **Cache Manager**: Manages key-value caches for Transformer layers
5. **Inference Engine**: Executes preemptive text generation with dual backend support (HuggingFace + Ollama)
6. **FastAPI Server**: Provides RESTful API with streaming support

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- macOS with Apple Silicon (M1/M2) for MPS support
- 8GB+ RAM recommended

### Installation

1. **Clone and setup**:
```bash
git clone <repository-url>
cd FastServe
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Start the server**:
```bash
python main.py
```

4. **Test the installation**:
```bash
# In another terminal
python examples/client_example.py
```

### Quick API Test

```bash
# Generate text
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "The future of AI is",
    "max_new_tokens": 50,
    "temperature": 0.7
  }'

# Check system status
curl "http://localhost:8000/health"
```

## 📚 Usage Examples

### Basic Text Generation

```python
import httpx
import asyncio

async def generate_text():
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8000/generate",
            json={
                "prompt": "Explain quantum computing in simple terms:",
                "max_new_tokens": 100,
                "temperature": 0.7,
                "top_p": 0.9
            }
        )
        result = response.json()
        print(f"Generated: {result['generated_text']}")
        print(f"Tokens: {result['total_tokens']}")
        print(f"Time: {result['completion_time']:.2f}s")

asyncio.run(generate_text())
```

### Using DeepSeek R1 14B (Local Ollama)

FastServe supports using your local DeepSeek model via Ollama for **much better performance**:

```python
from fastserve import FastServeServer
from fastserve.config import FastServeConfig

# Configure for DeepSeek
config = FastServeConfig(
    use_ollama=True,
    ollama_model="deepseek-r1:14b",
    max_new_tokens=200
)

server = FastServeServer(config)
# Now your server uses DeepSeek instead of DialoGPT!
```

**Prerequisites for DeepSeek:**
```bash
# 1. Install and start Ollama
ollama serve

# 2. Download DeepSeek model (if not already done)
ollama pull deepseek-r1:14b

# 3. Install additional dependency
pip install aiohttp>=3.8.0
```

**Benefits over default DialoGPT:**
- 🚀 **126x more parameters** (14.8B vs 117M)
- 📏 **131x larger context** (131K vs 1K tokens)  
- 🧠 **Advanced reasoning** and coding capabilities
- 🏠 **100% local** inference (no internet required)
- ⚡ **Better performance** on complex tasks

### Streaming Generation

```python
import httpx
import json

async def stream_generation():
    async with httpx.AsyncClient() as client:
        async with client.stream(
            "POST",
            "http://localhost:8000/generate/stream",
            json={"prompt": "Write a short story about robots:", "max_new_tokens": 200}
        ) as response:
            async for chunk in response.aiter_text():
                if chunk.startswith("data: "):
                    data = json.loads(chunk[6:])
                    if not data.get("is_final"):
                        print(data["token"], end="", flush=True)
```

## ⚙️ Configuration

### Command Line Options

```bash
# Model configuration
python main.py --model microsoft/DialoGPT-medium --max-length 2048

# Performance tuning
python main.py --device mps --cache-size 2048 --batch-size 4

# Scheduler configuration  
python main.py --num-queues 4 --queue-capacity 50

# Server configuration
python main.py --host 0.0.0.0 --port 8080 --workers 1
```

### Environment Configuration

The system automatically detects and optimizes for your hardware:

- **Mac M2/M1**: Uses MPS acceleration with optimized memory management
- **CPU fallback**: Automatic fallback for systems without MPS support
- **Memory optimization**: Dynamic cache sizing based on available system memory

## 📊 Monitoring & Statistics

### System Statistics Endpoint

```bash
curl "http://localhost:8000/stats/detailed"
```

Returns comprehensive metrics including:
- Queue statistics and job distribution
- Memory usage (GPU/CPU)
- Cache hit rates and performance
- Inference throughput and latency
- Preemption counts and effectiveness

### Real-time Monitoring

The server provides several monitoring endpoints:

- `/health` - Basic health check
- `/stats` - System overview
- `/stats/detailed` - Comprehensive statistics
- `/jobs/{job_id}` - Individual job status

## 🧪 Performance Benchmarks

Based on the original FastServe research, the system provides:

- **5.1x improvement** in average Job Completion Time (JCT)
- **6.4x improvement** in tail JCT compared to traditional systems
- **Efficient preemption** with minimal overhead per token
- **Memory efficiency** through intelligent cache management

### Mac M2 Performance

On Apple Silicon M2 with 16GB RAM:
- **Model loading**: ~10-30 seconds (depending on model size)
- **Inference latency**: ~50-200ms per token (model dependent)
- **Concurrent requests**: 4-8 simultaneous requests efficiently handled
- **Memory usage**: Optimized for unified memory architecture

## 🔧 Advanced Configuration

### Custom Model Configuration

```python
from fastserve.config import update_config

# Configure for larger models
update_config(
    model_name="microsoft/DialoGPT-large",
    max_sequence_length=2048,
    max_cache_size_mb=4096,
    gpu_memory_threshold=0.75
)
```

### Scheduler Tuning

```python
# Optimize for specific workloads
update_config(
    num_priority_queues=6,
    skip_threshold=0.8,  # More aggressive skipping
    promotion_threshold=3,  # Faster promotion
    demotion_threshold=15   # Slower demotion
)
```

## 🐛 Troubleshooting

### Common Issues

**1. MPS Not Available**
```bash
# Check MPS availability
python -c "import torch; print(torch.backends.mps.is_available())"
```

**2. Memory Issues**
```bash
# Reduce cache size
python main.py --cache-size 512 --batch-size 1
```

**3. Model Loading Errors**
```bash
# Use smaller model
python main.py --model microsoft/DialoGPT-small
```

**4. Connection Issues**
```bash
# Check server status
curl http://localhost:8000/health
```

## 📝 API Reference

### Generate Text (POST /generate)

**Request Body**:
```json
{
  "prompt": "string",
  "max_new_tokens": 100,
  "temperature": 0.7,
  "top_p": 0.9,
  "top_k": 50,
  "do_sample": true
}
```

**Response**:
```json
{
  "job_id": "uuid",
  "generated_text": "string", 
  "total_tokens": 42,
  "completion_time": 1.5,
  "queue_time": 0.1,
  "execution_time": 1.4,
  "preemption_count": 2
}
```

### Stream Generation (POST /generate/stream)

Returns Server-Sent Events with incremental tokens.

### System Statistics (GET /stats)

Returns system-wide performance metrics.

### Job Status (GET /jobs/{job_id})

Returns detailed status for a specific job.

## 🤝 Contributing

FastServe is based on academic research and implements the algorithms described in:

> "Fast Distributed Inference Serving for Large Language Models" by Bingyang Wu, Yinmin Zhong, Zili Zhang, Gang Huang, Xuanzhe Liu, and Xin Jin

The implementation focuses on single-machine optimization for Mac M2 systems while maintaining the core algorithmic innovations.

## 📄 License

This implementation is provided for educational and research purposes.

## 📞 Support

For issues, questions, or contributions:

1. Check the troubleshooting section
2. Review the examples in `examples/`
3. Monitor system statistics at `/stats/detailed`
4. Test with smaller models first

---

**Built for Mac M2 Apple Silicon** 🍎⚡
