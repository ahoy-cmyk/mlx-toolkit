# MLX Toolkit

🚀 **The ultimate toolkit for MLX-based LLM development on Apple Silicon**

**Simple. Beautiful. Powerful.** Everything you need to fine-tune, chat with, and manage LLMs on your Mac.

---

## ⚡ Quick Start

### 1. **Prerequisites (for fresh Mac setup)**
```bash
# Install Homebrew (if not already installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python 3.9+ and Git
brew install python3 git

# Install command line tools (if not already installed)  
xcode-select --install
```

**System Requirements:**
- **macOS 10.15+** (Big Sur or later recommended)
- **Apple Silicon** (M1/M2/M3/M4) for best performance
- **Python 3.9+** (Homebrew version recommended)
- **Git** for repository cloning
- **16GB+ RAM** recommended for 7B+ models
- **Command Line Tools** for compilation

### 2. **Install MLX Toolkit (once)**
```bash
git clone https://github.com/ahoy-cmyk/mlx-toolkit.git
cd mlx-toolkit
./install-mlx-toolkit.sh
```

**🚨 CRITICAL: Configure your PATH immediately after installation:**
```bash
# Add this line to your shell profile (.zshrc for zsh, .bash_profile for bash)
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc

# Reload your shell configuration
source ~/.zshrc

# Verify the command works
mlx-toolkit --help
```

**If you're using bash instead of zsh:**
```bash
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bash_profile
source ~/.bash_profile
```

### 3. **Create your first project**
```bash
mlx-toolkit init my-ai-project
cd my-ai-project
mlx-toolkit install-deps
```

### 4. **Download a model and start chatting**
```bash
mlx-toolkit download-model mlx-community/CodeLlama-7b-Instruct-hf-4bit-mlx-2
mlx-toolkit chat models/CodeLlama-7b-Instruct-hf-4bit-mlx-2
```

**That's it!** 🎉 You're now chatting with a state-of-the-art LLM on your Mac.

---

## 🌟 What Makes This Special

### 🎨 **Beautiful CLI Experience**
- **Live system dashboard** with real-time monitoring
- **Gorgeous model browser** with ratings and search  
- **Colorized chat interface** that's easy to follow
- **Rich progress bars** and visual feedback everywhere

### ⚡ **LoRA Fine-Tuning Made Easy**
- **One command** to fine-tune any quantized model
- **Intelligent checkpoint resumption** - auto-detects progress and runs exact remaining iterations
- **Live training monitoring** with ASCII loss curves and metrics
- **Smart data conversion** and validation splitting
- **Memory efficient** - train 8B models on 24GB RAM
- **Works with 4-bit models** that can't be fine-tuned normally

### 🛠️ **Zero Hassle Setup**
- **No manual venv activation** - everything automatic
- **Smart dependency management** per project
- **Isolated environments** - no conflicts between projects
- **Optimized for Apple Silicon** M1/M2/M3/M4 Macs

---

## 📚 Core Commands

| Command | What It Does |
|---------|-------------|
| `mlx-toolkit init <name>` | Create new project with isolated environment |
| `mlx-toolkit install-deps` | Install all MLX packages automatically |
| `mlx-toolkit status` | Beautiful system dashboard with live monitoring |
| `mlx-toolkit models` | Browse, search, and manage models |
| `mlx-toolkit download-model <id>` | Download models from Hugging Face |
| `mlx-toolkit create-training-data` | Generate properly formatted training data templates |
| `mlx-toolkit lora <model> <data>` | LoRA fine-tune quantized models with auto-resume |
| `mlx-toolkit chat <model>` | Colorized chat with any model |
| `mlx-toolkit test <model> <data>` | Evaluate model performance with comprehensive metrics |
| `mlx-toolkit train-monitor --live` | Live training dashboard with loss curves |
| `mlx-toolkit benchmark <model>` | Performance testing and speed analysis |
| `mlx-toolkit checkpoints` | Browse and manage training checkpoints |

---

## 🚀 Advanced Examples

### **Fine-tune a model with your data**
```bash
# Train a LoRA adapter (automatically resumes if interrupted!)
mlx-toolkit lora models/CodeLlama-7b-Instruct-hf-4bit-mlx-2 data/my-training-data.jsonl

# In another terminal, monitor training with live loss curves
mlx-toolkit train-monitor --live

# If training gets interrupted, just run the same command again - it will auto-resume!
# Automatically calculates exact remaining iterations (e.g., 400 total → 100 remaining)
# Disable auto-resume with: --no-resume

# Use custom output directory (resumption still works!)
mlx-toolkit lora models/CodeLlama-7b-Instruct-hf-4bit-mlx-2 data.jsonl --output /custom/path

# Browse your checkpoints (auto-detects all locations)
mlx-toolkit checkpoints

# Chat with your fine-tuned model
mlx-toolkit chat models/CodeLlama-7b-Instruct-hf-4bit-mlx-2 --adapter-path models/lora_adapters
```

### **Monitor your system in real-time**
```bash
# Live dashboard with animated charts
mlx-toolkit status --live

# Quick status snapshot
mlx-toolkit status
```

### **Browse and manage models**
```bash
# Beautiful model browser with ratings
mlx-toolkit models

# Search for specific models
mlx-toolkit models --search llama

# Show only downloaded models
mlx-toolkit models --filter downloaded
```

### **Performance monitoring and testing**
```bash
# Live training monitor with ASCII loss curves
mlx-toolkit train-monitor --live

# Quick training summary
mlx-toolkit train-monitor

# Benchmark model performance
mlx-toolkit benchmark models/CodeLlama-7b-Instruct-hf-4bit-mlx-2

# Test with custom prompt
mlx-toolkit benchmark models/CodeLlama-7b-Instruct-hf-4bit-mlx-2 --prompt "Write a Python function"

# Evaluate model with comprehensive metrics
mlx-toolkit test models/CodeLlama-7b-Instruct-hf-4bit-mlx-2 data/test-data.jsonl --metrics perplexity generation_quality
```

---

## 📊 Training Data Format

Your training data should be in JSONL format:

```jsonl
{"instruction": "What is machine learning?", "response": "Machine learning is..."}
{"instruction": "Explain neural networks", "response": "Neural networks are..."}
{"instruction": "How do I train a model?", "response": "To train a model..."}
```

**Need help getting started?** Generate templates automatically:
```bash
# Create sample training data with proper formatting
mlx-toolkit create-training-data --format instruction --samples 20
```

The toolkit automatically:
- ✅ **Converts** your data to MLX format
- ✅ **Splits** into 90% training, 10% validation  
- ✅ **Validates** data format and quality
- ✅ **Handles** both instruction-response and text formats

---

## 💡 Perfect for M4 Pro (24GB RAM)

**Recommended models:**
- `mlx-community/CodeLlama-7b-Instruct-hf-4bit-mlx-2` (4.0GB) ⭐⭐⭐⭐⭐
- `mlx-community/Llama-3.1-8B-Instruct-4bit` (4.9GB) ⭐⭐⭐⭐⭐
- `mlx-community/Mistral-7B-Instruct-v0.3-4bit` (4.1GB) ⭐⭐⭐⭐

**LoRA settings:**
- Use rank 8-16 for best results
- Batch size 4-8 works perfectly
- Can fine-tune 8B models easily

---

## 🎯 Real-World Workflow

```bash
# 1. Create a coding assistant project
mlx-toolkit init coding-assistant
cd coding-assistant
mlx-toolkit install-deps

# 2. Download a code model
mlx-toolkit download-model mlx-community/CodeLlama-7b-Instruct-hf-4bit-mlx-2

# 3. Check everything is ready
mlx-toolkit status

# 4. Fine-tune on your code examples
mlx-toolkit lora models/CodeLlama-7b-Instruct-hf-4bit-mlx-2 data/coding-examples.jsonl

# 5. In another terminal, monitor training progress
mlx-toolkit train-monitor --live

# 6. Chat with your personalized coding assistant
mlx-toolkit chat models/CodeLlama-7b-Instruct-hf-4bit-mlx-2 --adapter-path models/lora_adapters
```

---

## 🖥️ CLI Visual Experience

### **Status Dashboard**
```
🚀 MLX TOOLKIT STATUS DASHBOARD 🚀
┌─────────────────────────────────┬─────────────────────────────────┐
│ 🖥️  System Status               │ 📁 Project Status               │
│ 🔥 CPU    45.2%   ████████░░░░   │ 🔋 Virtual Environment          │
│ 💾 RAM    62.1%   ████████████░  │   ✅ Active                     │
│ 💽 Disk   34.8%   ██████░░░░░░░  │ 📦 MLX Packages                 │
│ 🐍 Python 3.11.5  ✅           │   ✅ mlx                        │
└─────────────────────────────────┤   ✅ mlx-lm                     │
                                  │   ✅ transformers               │
                                  │ 🤖 Models                       │
                                  │   📁 Llama-3.1-8B (4.9GB)      │
```

### **Model Browser**
```
🤖 MLX MODEL BROWSER 🤖
┌──────────────────────────┬──────┬───────┬────────────┬────────┬─────────────────┐
│ 🤖 Model                 │ Size │ Type  │ Status     │ Rating │ Actions         │
├──────────────────────────┼──────┼───────┼────────────┼────────┼─────────────────┤
│ Llama-3.1-8B-Instruct   │ 4.9GB│ Chat  │ ✅ Downloaded│ ⭐⭐⭐⭐⭐ │ 🗑️  Delete       │
│ Mistral-7B-Instruct     │ 4.1GB│ Chat  │ 📥 Available│ ⭐⭐⭐⭐☆ │ ⬇️  Download     │
│ CodeLlama-7b-Instruct   │ 4.0GB│ Code  │ 📥 Available│ ⭐⭐⭐⭐⭐ │ ⬇️  Download     │
└──────────────────────────┴──────┴───────┴────────────┴────────┴─────────────────┘
```

---

## 🔧 Troubleshooting

### **Fresh Mac Setup Issues**

**"Command not found: brew"**
```bash
# Install Homebrew first
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
# Follow the instructions to add Homebrew to PATH
```

**"xcrun: error: invalid active developer path"**
```bash
# Install Xcode Command Line Tools
xcode-select --install
```

**"python3: command not found"**
```bash
# Install Python via Homebrew
brew install python3

# Verify installation
python3 --version
```

### **MLX Toolkit Issues**

**🚨 "Command not found: mlx-toolkit" (MOST COMMON ISSUE)**

This happens because `~/.local/bin` is not in your PATH. Here's how to fix it:

**Step 1: Check which shell you're using**
```bash
echo $SHELL
# If output contains "zsh" → use .zshrc
# If output contains "bash" → use .bash_profile
```

**Step 2: Add to your shell profile**
```bash
# For zsh users (most common on modern Macs)
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc

# For bash users
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bash_profile
source ~/.bash_profile
```

**Step 3: Verify it's working**
```bash
# Check if the directory is in PATH
echo $PATH | grep -o ~/.local/bin

# Test the command
mlx-toolkit --help
```

**Still not working? Manual fix:**
```bash
# Open your shell profile in a text editor
nano ~/.zshrc  # or ~/.bash_profile for bash

# Add this line at the end:
export PATH="$HOME/.local/bin:$PATH"

# Save the file (Ctrl+X, then Y, then Enter in nano)
# Restart your terminal completely
```

**Package detection issues**
```bash
# Update dependencies in your project
mlx-toolkit install-deps

# Check status
mlx-toolkit status
```

### **Memory issues**
- Use 4-bit quantized models (recommended)
- Reduce LoRA batch size: `--batch-size 4`
- Use LoRA instead of full fine-tuning
- Close other applications

### **Model download fails**
```bash
# Login to Hugging Face if needed
huggingface-cli login

# Try a different model
mlx-toolkit models --search phi
```

---

## 🚀 What's New in Latest Version

### ✨ **Gorgeous CLI Experience**
- **Live status dashboard** with real-time system monitoring
- **Beautiful model browser** with search and filtering  
- **Enhanced chat interface** with color-coded conversations
- **Rich visual feedback** throughout all commands

### 🔥 **Epic Training & Performance Features**
- **Live training monitor** with ASCII loss curves and metrics
- **Comprehensive model benchmarking** with speed and memory analysis
- **Real-time training logs** automatically captured during LoRA fine-tuning
- **Performance comparison tables** with beautiful ratings

### 🔄 **Smart Checkpoint Management**
- **Automatic checkpoint detection** and resumption from interruptions
- **Beautiful checkpoint browser** with progress tracking and time estimates
- **Zero-effort recovery** - just rerun the same training command
- **Smart progress preservation** - never lose training work again

### ⚡ **LoRA Fine-Tuning**
- **Automatic data splitting** (90% train, 10% validation)
- **Smart data conversion** to MLX format
- **Memory efficient** training for quantized models
- **Real-time progress monitoring** with live loss visualization

### 🛠️ **Developer Experience**
- **Zero manual venv activation** - everything automatic
- **Better error messages** with helpful suggestions  
- **Comprehensive status monitoring** 
- **Simplified installation** and updates

---

## 🔄 Smart Checkpoint Management

**Revolutionary automatic training resumption** - Never lose training progress again! The toolkit intelligently handles interruptions and makes smart decisions about where to resume.

### **🧠 Intelligent Resumption**
The toolkit doesn't just resume - it makes **smart decisions** about the best checkpoint to use:

```bash
# Start training with 400 iterations
mlx-toolkit lora models/CodeLlama-7b-Instruct-hf-4bit-mlx-2 data/training.jsonl --iters 400

# Training gets interrupted at iteration 300...
# Just run the EXACT same command again!
mlx-toolkit lora models/CodeLlama-7b-Instruct-hf-4bit-mlx-2 data/training.jsonl --iters 400

# ✨ MAGIC HAPPENS:
# 🔍 Scans: finds 0000100, 0000200, 0000300, adapters.safetensors
# 🧠 Analyzes: "adapters.safetensors claims complete, but 0000300 is at 75% progress"
# 🎯 Decides: "Resume from 0000300_adapters.safetensors (300/400 iterations)"
# 🔧 Adjusts: "400 total → 100 remaining iterations"
# ⚡ Resumes: MLX-LM runs exactly 100 more iterations with pre-trained weights!
```

### **🔍 Advanced Checkpoint Browser**
Explore your training history with beautiful visualizations:

```bash
# Auto-detects ALL checkpoint directories across your project
mlx-toolkit checkpoints

# Browse specific custom location
mlx-toolkit checkpoints --path /my/custom/training/path

# Debug detection issues with detailed analysis
mlx-toolkit debug-checkpoints

# Debug specific directory with custom iteration target
mlx-toolkit debug-checkpoints /path/to/checkpoints --target-iters 500
```

### **🎯 Smart Detection Features**
- **🧠 Ambiguous State Handling**: When both final and numbered checkpoints exist, chooses the optimal resumption point
- **📊 Progress Analysis**: Shows completion percentage, time saved, and remaining work  
- **🔍 Multi-Format Support**: Detects `.safetensors` (MLX-LM standard) and `.npz` files
- **📁 Auto-Discovery**: Scans common locations and custom paths automatically
- **⚡ Filename Intelligence**: Parses `0000300_adapters.safetensors` → iteration 300

### **💡 Real-World Example**
```
🤔 Ambiguous Training State Detected

Found final checkpoint: adapters.safetensors (claims 400 iterations)
Latest numbered checkpoint: 0000300_adapters.safetensors (300 iterations)  
Target iterations: 400

🟢 Recommending resumption from numbered checkpoint (300 iterations)

⚡ Resumption Plan
Latest checkpoint: 0000300_adapters.safetensors
Progress: 300/400 iterations (75.0%)
Remaining: 100 iterations  
Time saved: ~75% of training time!
```

**What You Get:**
- 🔄 **Zero-effort resumption** - identical command, automatic detection
- 🧠 **Intelligent decisions** - chooses optimal checkpoint, not just latest file
- 🔧 **Perfect iteration counting** - automatically adjusts remaining iterations (e.g., 400 total → 100 remaining)
- 📊 **Complete progress tracking** - see exactly where you left off with visual progress bars
- 💾 **Bulletproof detection** - works with any output path or checkpoint directory
- 📈 **Beautiful browser** - rich tables showing all checkpoints with timestamps and sizes
- ⚡ **Accurate time estimates** - calculates exactly how much training time you save
- 🛡️ **Safety first** - never overwrites completed training or makes destructive decisions
- 🎯 **Custom path support** - works seamlessly with `--output /any/custom/path`

---

## 🔥 Training Monitor & Benchmarking

### **Live Training Dashboard**
Monitor your LoRA training in real-time with beautiful ASCII charts:

```bash
# Start training in one terminal
mlx-toolkit lora models/CodeLlama-7b-Instruct-hf-4bit-mlx-2 data/my-data.jsonl

# Monitor live in another terminal
mlx-toolkit train-monitor --live
```

**Features:**
- 📈 **Live loss curves** with smooth ASCII art visualization
- ⚡ **Real-time metrics** including speed, ETA, and loss velocity
- 🔥 **System monitoring** with CPU, RAM, and disk usage bars
- 🎯 **Smart file watching** - only updates when training log changes
- 📊 **Trend analysis** with improvement percentage tracking
- 🏆 **Best performance** tracking with step numbers

### **Model Benchmarking**
Test and compare model performance:

```bash
# Comprehensive performance testing
mlx-toolkit benchmark models/CodeLlama-7b-Instruct-hf-4bit-mlx-2

# Custom benchmarks
mlx-toolkit benchmark models/CodeLlama-7b-Instruct-hf-4bit-mlx-2 \
  --prompt "Write a sorting algorithm" \
  --iterations 20 \
  --max-tokens 200
```

**Benchmark Results Include:**
- ⚡ **Speed metrics** (tokens/sec, avg response time)
- 💾 **Memory analysis** (peak usage, efficiency ratings)
- 📊 **Performance ratings** with beautiful visual tables
- 🖥️ **System information** and optimization tips

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file.

## 👨‍💻 Author

**Stephan Arrington**
- GitHub: [@ahoy-cmyk](https://github.com/ahoy-cmyk)
- Specializing in Apple Silicon AI development tools

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

**Built with ❤️ for the MLX community by Stephan Arrington**