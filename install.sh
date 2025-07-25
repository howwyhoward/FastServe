#!/bin/bash
# FastServe Installation Script for Mac M2

echo "🚀 FastServe Installation Script"
echo "================================="

# Check if we're on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo "⚠️  This script is optimized for macOS. You may need to modify it for other systems."
fi

# Check Python version
echo "📋 Checking Python version..."
python_version=$(python3 --version 2>&1 | grep -oE '[0-9]+\.[0-9]+' | head -1)
required_version="3.8"

# Convert versions to comparable format (e.g., 3.13 -> 313, 3.8 -> 38)
version_num=$(echo $python_version | awk -F. '{printf "%d%02d", $1, $2}')
required_num=$(echo $required_version | awk -F. '{printf "%d%02d", $1, $2}')

if [[ $version_num -ge $required_num ]]; then
    echo "✅ Python $python_version found (>= $required_version required)"
else
    echo "❌ Python $python_version found, but >= $required_version required"
    echo "Please install Python 3.8 or higher"
    exit 1
fi

# Create and activate virtual environment
echo "🔧 Setting up virtual environment..."
if ! python3 -m venv fastserve_env; then
    echo "❌ Failed to create virtual environment"
    exit 1
fi

echo "📦 Installing Python dependencies in virtual environment..."
if source fastserve_env/bin/activate && pip install -r requirements.txt; then
    echo "✅ Dependencies installed successfully in virtual environment"
    echo ""
    echo "📝 Note: Virtual environment created at './fastserve_env/'"
    echo "   To activate: source fastserve_env/bin/activate"
    echo "   To deactivate: deactivate"
else
    echo "❌ Failed to install dependencies"
    echo "Alternative: Try running manually:"
    echo "  python3 -m venv fastserve_env"
    echo "  source fastserve_env/bin/activate" 
    echo "  pip install -r requirements.txt"
    exit 1
fi

# Test installation in virtual environment
echo "🧪 Testing installation..."
if source fastserve_env/bin/activate && python test_fastserve.py; then
    echo "
🎉 FastServe installation completed successfully!

Quick Start:
1. Activate environment: source fastserve_env/bin/activate
2. Start the server:     python main.py
3. Run examples:         python examples/client_example.py  (in new terminal)
4. Run benchmarks:       python examples/benchmark.py
5. Check health:         curl http://localhost:8000/health

For more options:        python main.py --help
Documentation:         README.md
"
else
    echo "❌ Installation test failed. Please check the errors above."
    exit 1
fi
