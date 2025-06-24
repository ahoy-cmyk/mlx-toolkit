"""Package management utilities for MLX Toolkit"""

import subprocess
import sys
import platform
from pathlib import Path
from rich.console import Console

console = Console()


class PackageManager:
    def __init__(self):
        self.is_mac = platform.system() == "Darwin"
        self.is_apple_silicon = self.is_mac and platform.machine() == "arm64"
    
    def install_mlx_packages(self, cuda=False, dev=False):
        """Install MLX and related packages"""
        if not self.is_mac:
            console.print("[yellow]Warning: MLX is optimized for Apple Silicon. Some features may not work on other platforms.[/]")
        
        packages = self._get_base_packages()
        
        if dev:
            packages.extend(self._get_dev_packages())
        
        try:
            # Upgrade pip first
            console.print("Upgrading pip...")
            subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], check=True)
            
            # Install packages
            console.print(f"Installing {len(packages)} packages...")
            
            # Install in batches to avoid issues
            batch_size = 5
            for i in range(0, len(packages), batch_size):
                batch = packages[i:i+batch_size]
                subprocess.run([sys.executable, "-m", "pip", "install"] + batch, check=True)
            
            # Verify MLX installation
            if self._verify_mlx_installation():
                console.print("[green]MLX installation verified successfully![/]")
            else:
                console.print("[yellow]Warning: MLX installation could not be verified[/]")
            
            return True
            
        except subprocess.CalledProcessError as e:
            console.print(f"[red]Error installing packages: {e}[/]")
            return False
    
    def _get_base_packages(self):
        """Get base package list"""
        packages = [
            # Core MLX packages
            "mlx",
            "mlx-lm",
            "huggingface-hub",
            "sentencepiece",
            
            # Essential ML packages
            "numpy",
            "transformers",
            "datasets",
            "tokenizers",
            "safetensors",
            "tqdm",
            "pyyaml",
            "pandas",
            
            # Visualization and notebooks
            "matplotlib",
            "seaborn",
            "jupyter",
            "ipykernel",
            
            # Additional utilities
            "accelerate",
            "protobuf",
            "fire",
        ]
        
        # Add torch for data preparation (CPU only for Mac)
        if self.is_mac:
            packages.append("torch")
        else:
            packages.append("torch")
            
        return packages
    
    def _get_dev_packages(self):
        """Get development packages"""
        return [
            "pytest",
            "pytest-cov",
            "black",
            "flake8",
            "mypy",
            "pre-commit",
            "ipdb",
            "wandb",
            "tensorboard",
        ]
    
    def _verify_mlx_installation(self):
        """Verify MLX is properly installed"""
        try:
            import mlx
            import mlx.core as mx
            
            # Try a simple operation
            a = mx.array([1.0, 2.0, 3.0])
            b = mx.array([4.0, 5.0, 6.0])
            c = a + b
            mx.eval(c)
            
            return True
        except Exception as e:
            console.print(f"[red]MLX verification failed: {e}[/]")
            return False
    
    def create_environment_file(self, project_path):
        """Create environment.yml for conda users"""
        env_content = """name: mlx-env
channels:
  - conda-forge
dependencies:
  - python=3.11
  - pip
  - numpy
  - pandas
  - matplotlib
  - jupyter
  - ipykernel
  - pip:
    - mlx>=0.18.0
    - mlx-lm>=0.18.0
    - transformers
    - datasets
    - tokenizers
    - huggingface-hub
    - safetensors
    - torch
    - accelerate
    - sentencepiece
    - fire
    - click
    - rich
    - tqdm
    - pyyaml
"""
        
        with open(project_path / "environment.yml", "w") as f:
            f.write(env_content)
    
    def check_system_requirements(self):
        """Check system requirements for MLX"""
        checks = {
            "Platform": platform.system(),
            "Architecture": platform.machine(),
            "Python Version": sys.version.split()[0],
            "MLX Compatible": self.is_apple_silicon,
        }
        
        console.print("[bold]System Requirements Check:[/]")
        for key, value in checks.items():
            status = "✓" if key != "MLX Compatible" or value else "⚠"
            console.print(f"{status} {key}: {value}")
        
        if not self.is_apple_silicon:
            console.print("\n[yellow]Note: MLX is optimized for Apple Silicon (M1/M2/M3/M4). Performance may be limited on other systems.[/]")
        
        return self.is_mac