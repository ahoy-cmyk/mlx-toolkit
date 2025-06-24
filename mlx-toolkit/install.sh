#!/bin/bash
# MLX Toolkit Installation Script

set -e

echo "🚀 MLX Toolkit Installer"
echo "========================"

# Check if running on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo "⚠️  Warning: MLX is optimized for macOS. Some features may not work on other platforms."
fi

# Check Python version
PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
PYTHON_VERSION_CHECK=$(python3 -c 'import sys; print(1 if sys.version_info >= (3, 9) else 0)')

if [[ "$PYTHON_VERSION_CHECK" -eq 0 ]]; then
    echo "❌ Error: Python 3.9 or higher is required (found $PYTHON_VERSION)"
    exit 1
fi

echo "✓ Python $PYTHON_VERSION found"

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install MLX Toolkit
echo "Installing MLX Toolkit..."
pip install -e .

# Create mlx-toolkit command
echo "Setting up mlx-toolkit command..."
chmod +x mlx_toolkit/cli.py

echo ""
echo "✅ Installation complete!"
echo ""
echo "To get started:"
echo "1. Activate the virtual environment: source venv/bin/activate"
echo "2. Run: mlx-toolkit --help"
echo ""
echo "Quick start:"
echo "  mlx-toolkit init my-project    # Create a new project"
echo "  cd my-project"
echo "  source venv/bin/activate"
echo "  mlx-toolkit install-deps       # Install MLX dependencies"
echo ""