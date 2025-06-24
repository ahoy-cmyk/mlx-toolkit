#!/bin/bash
# MLX Toolkit Global Installer
# This installs the toolkit globally so you can use 'mlx-toolkit' from anywhere

set -e

echo "🚀 MLX Toolkit Global Installer"
echo "================================"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if running on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo -e "${YELLOW}⚠️  Warning: MLX is optimized for macOS. Some features may not work on other platforms.${NC}"
fi

# Check Python version
echo "Checking Python version..."
PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
PYTHON_VERSION_CHECK=$(python3 -c 'import sys; print(1 if sys.version_info >= (3, 9) else 0)')

if [[ "$PYTHON_VERSION_CHECK" -eq 0 ]]; then
    echo -e "${RED}❌ Error: Python 3.9 or higher is required (found $PYTHON_VERSION)${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Python $PYTHON_VERSION found${NC}"

# Set installation directory
INSTALL_DIR="$HOME/.mlx-toolkit"
BIN_DIR="$HOME/.local/bin"

echo "Installing MLX Toolkit to: $INSTALL_DIR"

# Create directories
mkdir -p "$INSTALL_DIR"
mkdir -p "$BIN_DIR"

# Copy toolkit files
echo "Copying toolkit files..."
cp -r mlx-toolkit/* "$INSTALL_DIR/"

# Create virtual environment for the toolkit
echo "Creating toolkit virtual environment..."
cd "$INSTALL_DIR"
python3 -m venv toolkit-env

# Activate and install
echo "Installing toolkit dependencies..."
source toolkit-env/bin/activate
pip install --upgrade pip
pip install -e .

# Create global command script
echo "Creating global mlx-toolkit command..."
cat > "$BIN_DIR/mlx-toolkit" << 'EOF'
#!/bin/bash
# MLX Toolkit Global Command

# Get the directory where this script is located
SCRIPT_DIR="$HOME/.mlx-toolkit"

# Check if toolkit is installed
if [[ ! -d "$SCRIPT_DIR/toolkit-env" ]]; then
    echo "❌ MLX Toolkit not found. Please reinstall."
    exit 1
fi

# Activate toolkit environment and run command
source "$SCRIPT_DIR/toolkit-env/bin/activate"
python -m mlx_toolkit.cli "$@"
EOF

chmod +x "$BIN_DIR/mlx-toolkit"

# Check if ~/.local/bin is in PATH
if [[ ":$PATH:" != *":$HOME/.local/bin:"* ]]; then
    echo ""
    echo -e "${RED}🚨 CRITICAL: You MUST add ~/.local/bin to your PATH!${NC}"
    echo -e "${YELLOW}Without this step, 'mlx-toolkit' command will NOT work.${NC}"
    echo ""
    echo "Choose your shell and run ONE of these commands:"
    echo ""
    echo -e "${GREEN}For zsh users (most modern Macs):${NC}"
    echo "echo 'export PATH=\"\$HOME/.local/bin:\$PATH\"' >> ~/.zshrc && source ~/.zshrc"
    echo ""
    echo -e "${GREEN}For bash users:${NC}"
    echo "echo 'export PATH=\"\$HOME/.local/bin:\$PATH\"' >> ~/.bash_profile && source ~/.bash_profile"
    echo ""
    echo -e "${YELLOW}Not sure which shell? Run: echo \$SHELL${NC}"
    echo ""
    echo -e "${GREEN}After running the command above, test with: mlx-toolkit --help${NC}"
fi

# Create update script
echo "Creating update script..."
cat > "$BIN_DIR/mlx-toolkit-update" << 'EOF'
#!/bin/bash
# MLX Toolkit Update Script

echo "🔄 Updating MLX Toolkit..."

# Check if we're in the right directory
if [[ ! -f "install-mlx-toolkit.sh" ]]; then
    echo "❌ Error: Run this from the mlx-toolkit repository directory"
    echo "Or git pull the latest changes first"
    exit 1
fi

# Run the installer to update
echo "Running installer to update..."
./install-mlx-toolkit.sh

echo "✅ MLX Toolkit updated successfully!"
EOF

chmod +x "$BIN_DIR/mlx-toolkit-update"

echo ""
echo -e "${GREEN}✅ MLX Toolkit installed successfully!${NC}"
echo ""
echo "🎯 Quick Start:"
echo "1. Make sure ~/.local/bin is in your PATH (see above if needed)"
echo "2. Create your first project:"
echo "   mlx-toolkit init my-ai-project"
echo "3. Follow the prompts!"
echo ""
echo "📚 Commands:"
echo "   mlx-toolkit --help           # Show help"
echo "   mlx-toolkit-update           # Update the toolkit"
echo ""
echo "🔄 To update later:"
echo "   cd /path/to/mlx-toolkit-repo"
echo "   git pull"
echo "   mlx-toolkit-update"
echo ""
echo "🗑️  To uninstall:"
echo "   rm -rf ~/.mlx-toolkit ~/.local/bin/mlx-toolkit ~/.local/bin/mlx-toolkit-update"
echo ""