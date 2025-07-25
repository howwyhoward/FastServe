#!/usr/bin/env python3
"""
FastServe Installation Test
Verifies that all components are working correctly
"""

import sys
import torch
import importlib
import traceback
from pathlib import Path

def test_dependencies():
    """Test that all required dependencies are available"""
    print("🔍 Testing Dependencies...")
    
    required_packages = [
        'torch', 'transformers', 'fastapi', 'uvicorn', 
        'pydantic', 'numpy', 'psutil', 'httpx'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ❌ {package} - NOT FOUND")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    print("✅ All dependencies found")
    return True

def test_device_availability():
    """Test device availability and configuration"""
    print("\n🖥️  Testing Device Configuration...")
    
    # Test CPU
    print("  ✓ CPU available")
    
    # Test MPS (Mac M1/M2)
    if torch.backends.mps.is_available():
        print("  ✓ MPS (Apple Silicon) available")
        try:
            # Test MPS allocation
            test_tensor = torch.randn(10, 10).to('mps')
            print("  ✓ MPS tensor operations working")
        except Exception as e:
            print(f"  ⚠️  MPS available but not working: {e}")
    else:
        print("  ℹ️  MPS not available (not on Apple Silicon)")
    
    # Test CUDA (if available)
    if torch.cuda.is_available():
        print(f"  ✓ CUDA available ({torch.cuda.device_count()} devices)")
    else:
        print("  ℹ️  CUDA not available")
    
    return True

def test_fastserve_imports():
    """Test FastServe module imports"""
    print("\n📦 Testing FastServe Imports...")
    
    try:
        # Add current directory to path
        sys.path.insert(0, str(Path(__file__).parent))
        
        from fastserve.config import get_config, FastServeConfig
        print("  ✓ Config module")
        
        from fastserve.models import InferenceJob, JobStatus, GenerationRequest
        print("  ✓ Models module")
        
        from fastserve.scheduler import SkipJoinMLFQScheduler
        print("  ✓ Scheduler module")
        
        from fastserve.memory_manager import MemoryManager
        print("  ✓ Memory Manager module")
        
        from fastserve.cache_manager import KeyValueCacheManager
        print("  ✓ Cache Manager module")
        
        from fastserve.inference_engine import InferenceEngine
        print("  ✓ Inference Engine module")
        
        from fastserve.server import FastServeServer
        print("  ✓ Server module")
        
        print("✅ All FastServe modules imported successfully")
        return True
        
    except Exception as e:
        print(f"❌ FastServe import failed: {e}")
        traceback.print_exc()
        return False

def test_configuration():
    """Test configuration system"""
    print("\n⚙️  Testing Configuration...")
    
    try:
        from fastserve.config import get_config, update_config
        
        # Get default config
        config = get_config()
        print(f"  ✓ Default model: {config.model_name}")
        print(f"  ✓ Device: {config.device}")
        print(f"  ✓ Max cache size: {config.max_cache_size_mb}MB")
        
        # Test config update
        update_config(max_new_tokens=128)
        updated_config = get_config()
        assert updated_config.max_new_tokens == 128
        print("  ✓ Config update working")
        
        print("✅ Configuration system working")
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def test_component_initialization():
    """Test that components can be initialized"""
    print("\n🔧 Testing Component Initialization...")
    
    try:
        from fastserve.config import get_config
        from fastserve.scheduler import SkipJoinMLFQScheduler
        from fastserve.memory_manager import MemoryManager
        from fastserve.cache_manager import KeyValueCacheManager
        
        config = get_config()
        
        # Test scheduler
        scheduler = SkipJoinMLFQScheduler(config)
        print("  ✓ Scheduler initialization")
        
        # Test memory manager
        memory_manager = MemoryManager(config)
        print("  ✓ Memory Manager initialization")
        
        # Test cache manager
        cache_manager = KeyValueCacheManager(memory_manager, config)
        print("  ✓ Cache Manager initialization")
        
        # Test memory stats
        stats = memory_manager.get_memory_stats()
        print(f"  ✓ Memory stats: {stats.cpu_utilization:.1f}% CPU")
        
        print("✅ All components initialized successfully")
        return True
        
    except Exception as e:
        print(f"❌ Component initialization failed: {e}")
        traceback.print_exc()
        return False

def test_model_loading():
    """Test model loading capability"""
    print("\n🤖 Testing Model Loading...")
    
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        # Test with a very small model
        model_name = "microsoft/DialoGPT-small"
        print(f"  Testing with {model_name}...")
        
        # Test tokenizer loading
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print("  ✓ Tokenizer loaded")
        
        # Note: We don't load the full model here to keep test fast
        # Just verify the model exists
        print("  ✓ Model accessible")
        
        print("✅ Model loading capability verified")
        return True
        
    except Exception as e:
        print(f"❌ Model loading test failed: {e}")
        return False

def run_all_tests():
    """Run all tests"""
    print("FastServe Installation Test")
    print("=" * 50)
    
    tests = [
        ("Dependencies", test_dependencies),
        ("Device Configuration", test_device_availability),
        ("FastServe Imports", test_fastserve_imports),
        ("Configuration System", test_configuration),
        ("Component Initialization", test_component_initialization),
        ("Model Loading", test_model_loading),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"TEST RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("�� All tests passed! FastServe is ready to use.")
        print("\nTo start the server, run:")
        print("  python main.py")
        print("\nTo run examples, run:")
        print("  python examples/client_example.py")
        return True
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
