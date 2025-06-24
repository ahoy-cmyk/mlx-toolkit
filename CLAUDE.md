# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains the MLX Toolkit - an all-in-one toolkit for MLX-based LLM development on Apple Silicon Macs. The toolkit provides a comprehensive CLI for managing MLX projects, models, fine-tuning, and inference with **intelligent checkpoint resumption** and **live training monitoring**.

**Author**: Stephan Arrington (@ahoy-cmyk)  
**Repository**: https://github.com/ahoy-cmyk/mlx-toolkit

## Key Commands

### Global Installation (One-Time Setup)
```bash
# Install toolkit globally
./install-mlx-toolkit.sh

# CRITICAL: Configure PATH after installation
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc && source ~/.zshrc

# Update toolkit
git pull
mlx-toolkit-update

# Development mode (for contributing)
cd mlx-toolkit
pip install -e . # in toolkit's own venv
```

### Project Management
```bash
# Create isolated project
mlx-toolkit init <project-name>
cd <project-name>
mlx-toolkit install-deps  # Installs mlx, mlx-lm, huggingface-hub, sentencepiece

# Check system status
mlx-toolkit status --live
```

### Model Operations
```bash
# Browse and download models
mlx-toolkit models --search llama
mlx-toolkit download-model mlx-community/Llama-3.1-8B-Instruct-4bit

# Create training data templates
mlx-toolkit create-training-data --format instruction --samples 50

# Fine-tune with intelligent checkpoint resumption
mlx-toolkit lora models/Llama-3.1-8B-Instruct-4bit data/training.jsonl --iters 1000

# Monitor training in real-time (separate terminal)
mlx-toolkit train-monitor --live

# Browse checkpoints and manage resumption
mlx-toolkit checkpoints

# Interactive chat with models
mlx-toolkit chat models/Llama-3.1-8B-Instruct-4bit

# Evaluate models with comprehensive metrics
mlx-toolkit test models/Llama-3.1-8B-Instruct-4bit data/test.jsonl --metrics perplexity generation_quality

# Benchmark model performance
mlx-toolkit benchmark models/Llama-3.1-8B-Instruct-4bit --iterations 20
```

## Architecture

### Project Structure
```
ai-tools/
├── README.md                    # Parent repository overview
├── LICENSE                      # MIT License
├── CONTRIBUTING.md              # Contribution guidelines
├── install-mlx-toolkit.sh       # Global installer script
└── mlx-toolkit/                 # Main toolkit directory
    ├── README.md                # Detailed toolkit documentation
    ├── setup.py                 # Python package setup
    ├── requirements.txt         # Base dependencies
    ├── install.sh               # Local installer
    ├── mlx_toolkit/             # Main package
    │   ├── __init__.py
    │   ├── cli.py               # Main CLI with all commands (3500+ lines)
    │   ├── project_manager.py   # Project creation and management
    │   ├── package_manager.py   # MLX package installation & system checks
    │   └── model_manager.py     # Model operations (download, management)
    ├── templates/               # Script templates for advanced usage
    │   ├── finetune_script.py   # Advanced fine-tuning script
    │   ├── test_model.py        # Model evaluation script
    │   └── query_model.py       # Interactive query tool
    └── scripts/                 # Utility scripts
```

### Key Components

1. **CLI (cli.py)**: Comprehensive command-line interface with 17 commands using Click + Rich
2. **ProjectManager**: Handles project creation, virtual environment setup, and isolation
3. **PackageManager**: Manages MLX installation, system requirements, Apple Silicon optimization
4. **ModelManager**: Handles model downloads from HuggingFace with proper model management

### Core CLI Commands

**Essential Commands:**
- `init` - Create new project with isolated environment
- `install-deps` - Install all MLX packages automatically
- `status` - Beautiful system dashboard with live monitoring
- `models` - Browse, search, and manage models
- `download-model` - Download models from Hugging Face
- `lora` - LoRA fine-tune with intelligent checkpoint resumption
- `chat` - Colorized chat interface with any model

**Advanced Commands:**
- `create-training-data` - Generate properly formatted training data templates
- `test` - Evaluate model performance with comprehensive metrics
- `train-monitor` - Live training dashboard with ASCII loss curves
- `benchmark` - Performance testing and speed analysis
- `checkpoints` - Browse and manage training checkpoints
- `debug-checkpoints` - Debug checkpoint detection (troubleshooting)

## Revolutionary Checkpoint Management

### Intelligent Resumption System
The toolkit features **smart checkpoint detection** that:

1. **Auto-detects** interrupted training from safetensors files (`0000300_adapters.safetensors`)
2. **Calculates remaining iterations** (e.g., 400 total - 300 done = 100 remaining)
3. **Adjusts MLX-LM command** to run exactly the right number of iterations
4. **Handles ambiguous states** when both final and numbered checkpoints exist
5. **Works with custom paths** and multiple checkpoint formats

### Key Implementation Details
- **File**: `cli.py` lines 3306-3528 (`_detect_and_handle_checkpoints` function)
- **Smart logic**: Chooses optimal checkpoint for resumption, not just latest file
- **Iteration adjustment**: Modifies `--iters` parameter based on remaining work
- **Progress tracking**: Shows completion percentage and time saved
- **Format support**: MLX-LM `.safetensors` format with numbered checkpoints

## Important Design Decisions

### Architecture Principles
- **Global vs Project Separation**: Toolkit installed globally (~/.mlx-toolkit), projects are isolated
- **PATH Management**: Critical installation step - toolkit installs to ~/.local/bin
- **Virtual Environment Isolation**: Each project has independent virtual environment
- **Apple Silicon Optimized**: Designed for M1/M2/M3/M4 with memory considerations

### Technology Stack
- **Core MLX Stack**: mlx, mlx-lm, huggingface-hub, sentencepiece
- **CLI Framework**: Click + Rich for beautiful terminal experience
- **Progress Monitoring**: Live ASCII charts and real-time metrics
- **Model Format**: Safetensors (MLX-LM standard) with automatic quantization
- **Training Method**: LoRA adaptation for memory-efficient fine-tuning

### User Experience Focus
- **Zero manual venv activation** - everything automatic
- **Beautiful visual feedback** - Rich console with colors, progress bars, tables
- **Comprehensive error handling** - helpful messages with suggested solutions
- **Fresh Mac support** - Complete prerequisites (Homebrew, Python, Git, Xcode tools)

## System Requirements

**Essential Prerequisites:**
- macOS 10.15+ (Big Sur or later recommended)
- Apple Silicon (M1/M2/M3/M4) for optimal performance
- Python 3.9+ (Homebrew installation recommended)
- Git for repository management
- Xcode Command Line Tools
- 16GB+ RAM recommended for 7B+ models

**Installation Dependencies:**
```bash
# Install via Homebrew
brew install python3 git
xcode-select --install
```

## Development Guidelines

### Code Quality Standards
- **Error Handling**: Comprehensive exception handling throughout all modules
- **User Feedback**: Use Rich console for consistent, beautiful output
- **Apple Silicon Checks**: Always verify compatibility in PackageManager
- **Template Scripts**: Self-contained and runnable for user customization

### CLI Development
- **Command Structure**: Follow existing pattern with Click decorators
- **Help Text**: Comprehensive descriptions with examples
- **Progress Feedback**: Use Rich progress bars for long operations
- **Error Messages**: Actionable suggestions for problem resolution

### Checkpoint System Development
- **File Detection**: Support multiple patterns (adapters*.safetensors, 0000xxx_adapters.safetensors)
- **State Analysis**: Smart logic to handle ambiguous completion states
- **MLX Integration**: Proper integration with MLX-LM's resumption parameters
- **User Communication**: Clear progress indication and time estimates

## Technical Notes

### MLX-Specific Considerations
- **Platform Requirement**: MLX requires macOS and is optimized for Apple Silicon
- **Memory Management**: Use 4-bit quantized models for optimal performance
- **Training Format**: LoRA adaptation with safetensors checkpoints
- **Model Loading**: Automatic conversion and quantization when downloaded

### PATH Configuration Critical Issue
The most common user issue is PATH configuration. The toolkit:
1. **Installs to**: `~/.local/bin/mlx-toolkit`
2. **Requires**: `export PATH="$HOME/.local/bin:$PATH"` in shell profile
3. **Auto-detects**: Shell type (zsh vs bash) and provides specific instructions
4. **Verification**: Multiple methods to test and troubleshoot PATH issues

### Testing and Verification
- **Installation**: Test complete flow from fresh Mac to working toolkit
- **Commands**: Verify all 17 CLI commands work with proper error handling
- **Resumption**: Test checkpoint detection with various training scenarios
- **Performance**: Benchmark on different Apple Silicon models (M1-M4)

## Common Issues and Solutions

### PATH Issues (Most Common)
```bash
# Check shell type
echo $SHELL

# Add to appropriate profile
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc && source ~/.zshrc
```

### Fresh Mac Setup
```bash
# Install prerequisites
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
brew install python3 git
xcode-select --install
```

### Memory Issues
- Use 4-bit quantized models (recommended)
- Reduce LoRA batch size: `--batch-size 4`
- Close other applications during training

This toolkit represents a production-ready, professional-grade solution for MLX-based AI development on Apple Silicon.