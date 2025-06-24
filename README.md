# AI Tools

🚀 **Collection of AI development tools optimized for Apple Silicon**

## MLX Toolkit

**All-in-one toolkit for MLX-based LLM development on Apple Silicon Macs**

### 📋 Prerequisites

**Essential requirements for a fresh Mac:**
```bash
# Install Homebrew (if not already installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python 3.9+ and Git
brew install python3 git

# Install command line tools (if not already installed)
xcode-select --install
```

**System Requirements:**
- **macOS** (10.15+ recommended) 
- **Apple Silicon** (M1/M2/M3/M4) for optimal performance
- **Python 3.9+** (installed via Homebrew recommended)
- **Git** for repository management
- **16GB+ RAM** recommended for larger models

### ⚡ Quick Install

```bash
git clone https://github.com/ahoy-cmyk/mlx-toolkit.git
cd mlx-toolkit  
./install-mlx-toolkit.sh
```

**🚨 IMPORTANT: After installation, you MUST add to your PATH:**
```bash
# Add this line to your shell profile
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc

# Reload your shell (or restart terminal)
source ~/.zshrc

# Test it works
mlx-toolkit --help
```

### ✨ What You Get

- **Global `mlx-toolkit` command** - works from anywhere
- **Project isolation** - each project has its own environment  
- **Complete MLX stack** - mlx, mlx-lm, huggingface-hub, sentencepiece
- **Intelligent checkpoint resumption** - never lose training progress again
- **Live training monitoring** - beautiful ASCII charts and real-time metrics
- **Easy workflows** - download, fine-tune, chat with models in minutes

### 🎯 30-Second Example

```bash
# After installation, FIRST add to PATH:
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc && source ~/.zshrc

# Now you can use mlx-toolkit:
mlx-toolkit init my-llm-project
cd my-llm-project
mlx-toolkit install-deps
mlx-toolkit download-model mlx-community/Llama-3.1-8B-Instruct-4bit
mlx-toolkit chat models/Llama-3.1-8B-Instruct-4bit  # Chat with your model!

# Fine-tune with automatic resumption
mlx-toolkit create-training-data --format instruction --samples 50
mlx-toolkit lora models/Llama-3.1-8B-Instruct-4bit training_data.jsonl
```

**✨ No venv activation needed - it's all automatic!**

[**📚 Full Documentation →**](./mlx-toolkit/README.md)

---

## 🏗️ Architecture

- **Clean separation**: Toolkit is global, projects are isolated
- **Easy updates**: `mlx-toolkit-update` keeps you current
- **No conflicts**: Each project has its own MLX environment
- **Smart resumption**: Automatically detects and resumes interrupted training
- **M4 Pro optimized**: Perfect for 24GB M4 Pro workflows

## 📄 License

MIT License - see [LICENSE](LICENSE)

## 👨‍💻 Author

**Stephan Arrington**
- GitHub: [@ahoy-cmyk](https://github.com/ahoy-cmyk)
- MLX Toolkit: Advanced AI development tools for Apple Silicon

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md)