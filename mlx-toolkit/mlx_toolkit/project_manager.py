"""Project management utilities for MLX Toolkit"""

import os
import subprocess
import shutil
from pathlib import Path
from rich.console import Console

console = Console()


class ProjectManager:
    def __init__(self):
        self.templates_dir = Path(__file__).parent.parent / "templates"
    
    def create_project(self, project_name, python="python3", template="basic"):
        """Create a new MLX project with virtual environment"""
        project_path = Path(project_name)
        
        if project_path.exists():
            console.print(f"[red]Error: Directory '{project_name}' already exists[/]")
            return False
        
        try:
            # Create project directory
            project_path.mkdir(parents=True)
            
            # Create virtual environment
            console.print(f"Creating virtual environment...")
            subprocess.run([python, "-m", "venv", project_path / "venv"], check=True)
            
            # Create project structure
            self._create_project_structure(project_path, template)
            
            # Create initial requirements.txt
            self._create_requirements_file(project_path)
            
            # Create .gitignore
            self._create_gitignore(project_path)
            
            # Create README
            self._create_readme(project_path, project_name)
            
            return True
            
        except Exception as e:
            console.print(f"[red]Error creating project: {e}[/]")
            if project_path.exists():
                shutil.rmtree(project_path)
            return False
    
    def _create_project_structure(self, project_path, template):
        """Create the project directory structure"""
        dirs = [
            "src",
            "data",
            "models",
            "scripts",
            "tests",
            "configs",
        ]
        
        if template == "advanced":
            dirs.extend([
                "notebooks",
                "experiments",
                "logs",
                "checkpoints",
            ])
        
        for dir_name in dirs:
            (project_path / dir_name).mkdir(exist_ok=True)
            
        # Create __init__.py files
        (project_path / "src" / "__init__.py").touch()
        (project_path / "tests" / "__init__.py").touch()
    
    def _create_requirements_file(self, project_path):
        """Create initial requirements.txt"""
        requirements = [
            "mlx>=0.18.0",
            "mlx-lm>=0.18.0",
            "numpy",
            "transformers",
            "datasets",
            "tokenizers",
            "torch",  # For dataset preparation
            "tqdm",
            "pyyaml",
            "pandas",
            "matplotlib",
            "seaborn",
            "jupyter",
            "ipykernel",
        ]
        
        with open(project_path / "requirements.txt", "w") as f:
            f.write("\n".join(requirements))
    
    def _create_gitignore(self, project_path):
        """Create .gitignore file"""
        gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
env/
ENV/

# MLX/ML
models/
*.gguf
*.safetensors
*.bin
checkpoints/
logs/
wandb/

# Data
data/raw/
data/processed/
*.csv
*.json
*.jsonl
*.parquet

# Jupyter
.ipynb_checkpoints/
*.ipynb_checkpoints

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# Project
.env
.env.local
"""
        
        with open(project_path / ".gitignore", "w") as f:
            f.write(gitignore_content)
    
    def _create_readme(self, project_path, project_name):
        """Create README.md"""
        readme_content = f"""# {project_name}

MLX-based LLM project created with MLX Toolkit.

## 🚀 Super Simple Setup

**Just `cd` into this directory and run commands. No virtual environment activation needed!**

### 1. Install MLX Dependencies (First Time Only)
```bash
cd {project_name}
mlx-toolkit install-deps
```

### 2. Start Building!
```bash
# Download a model
mlx-toolkit download-model microsoft/phi-2

# Chat with it immediately
mlx-toolkit query models/phi-2 -i

# Fine-tune it
mlx-toolkit finetune models/phi-2 data/train.jsonl
```

## ✨ That's It!

The toolkit automatically handles all the virtual environment stuff behind the scenes. You just run commands from this project directory.

## 📁 Project Structure

- `src/`: Your custom code
- `data/`: Training and test datasets  
- `models/`: Downloaded and fine-tuned models
- `scripts/`: Utility scripts
- `configs/`: Configuration files
- `venv/`: Auto-managed Python environment (don't touch this!)

## 🔄 Daily Workflow

```bash
# 1. Enter project directory
cd {project_name}

# 2. Use any mlx-toolkit command
mlx-toolkit download-model microsoft/phi-2
mlx-toolkit query models/phi-2 -i
mlx-toolkit finetune models/phi-2 data/train.jsonl
```

## 📦 What's Included in Your Environment

- **mlx**: Apple Silicon optimized ML framework
- **mlx-lm**: Language model support
- **huggingface-hub**: Model downloading
- **sentencepiece**: Tokenization
- **transformers, datasets, tokenizers**: ML ecosystem
- **jupyter**: Notebooks for experimentation

## 💡 Pro Tips

- All commands automatically use this project's isolated environment
- You can have multiple projects without conflicts
- No need to remember virtual environment commands
- Just `cd` to project and run `mlx-toolkit` commands!
"""
        
        with open(project_path / "README.md", "w") as f:
            f.write(readme_content)