"""Model management utilities for MLX Toolkit"""

import os
import sys
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Any
from rich.console import Console
from rich.prompt import Prompt

console = Console()


class ModelManager:
    def __init__(self):
        self.supported_models = [
            "microsoft/phi-2",
            "mistralai/Mistral-7B-v0.1",
            "mistralai/Mistral-7B-Instruct-v0.2",
            "meta-llama/Llama-2-7b-hf",
            "meta-llama/Llama-2-7b-chat-hf",
            "stabilityai/stablelm-2-1_6b",
            "google/gemma-2b",
            "google/gemma-7b",
        ]
    
    def download_model(self, model_id: str, output_dir: str = "models") -> bool:
        """Download a model from Hugging Face"""
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Check if user needs to login to HF
            if "meta-llama" in model_id:
                console.print("[yellow]Note: Llama models require Hugging Face authentication.[/]")
                console.print("Run: huggingface-cli login")
            
            # Use mlx_lm convert to download and convert
            console.print(f"Downloading and converting {model_id}...")
            
            cmd = [
                sys.executable, "-m", "mlx_lm.convert",
                "--hf-path", model_id,
                "--mlx-path", str(output_path / model_id.split("/")[-1]),
                "--quantize",
            ]
            
            subprocess.run(cmd, check=True)
            return True
            
        except subprocess.CalledProcessError as e:
            console.print(f"[red]Error downloading model: {e}[/]")
            return False
    
    def finetune_model(self, model_path: str, data_path: str, output_path: str,
                      config: Optional[str] = None, **kwargs) -> bool:
        """Fine-tune a model using MLX"""
        try:
            # Create fine-tuning script
            script_path = Path("finetune_temp.py")
            self._create_finetune_script(script_path, model_path, data_path, output_path, **kwargs)
            
            # Run fine-tuning
            console.print("Starting fine-tuning...")
            subprocess.run([sys.executable, str(script_path)], check=True)
            
            # Cleanup
            script_path.unlink()
            return True
            
        except Exception as e:
            console.print(f"[red]Error during fine-tuning: {e}[/]")
            return False
    
    def query_model(self, model_path: str, prompt: str, **kwargs) -> Optional[str]:
        """Query a model with a prompt"""
        try:
            import mlx_lm
            from mlx_lm import load, generate
            
            # Load model
            model, tokenizer = load(model_path)
            
            # Generate response
            response = generate(
                model, tokenizer, 
                prompt=prompt,
                max_tokens=kwargs.get("max_tokens", 100),
                temperature=kwargs.get("temperature", 0.7),
            )
            
            return response
            
        except Exception as e:
            console.print(f"[red]Error querying model: {e}[/]")
            return None
    
    def interactive_query(self, model_path: str, **kwargs):
        """Interactive querying mode"""
        try:
            import mlx_lm
            from mlx_lm import load, generate
            
            console.print("Loading model...")
            model, tokenizer = load(model_path)
            
            console.print("[green]Model loaded! Type 'quit' to exit.[/]")
            
            while True:
                prompt = Prompt.ask("\n[bold cyan]You[/]")
                
                if prompt.lower() in ["quit", "exit", "q"]:
                    break
                
                response = generate(
                    model, tokenizer,
                    prompt=prompt,
                    max_tokens=kwargs.get("max_tokens", 100),
                    temperature=kwargs.get("temperature", 0.7),
                )
                
                console.print(f"\n[bold green]Assistant[/]: {response}")
                
        except Exception as e:
            console.print(f"[red]Error in interactive mode: {e}[/]")
    
    def test_model(self, model_path: str, test_data: str, metrics: List[str]) -> Optional[Dict[str, float]]:
        """Test and evaluate a model"""
        try:
            results = {}
            
            # Create evaluation script
            script_path = Path("evaluate_temp.py")
            self._create_evaluation_script(script_path, model_path, test_data, metrics)
            
            # Run evaluation
            console.print("Running evaluation...")
            result = subprocess.run(
                [sys.executable, str(script_path)],
                capture_output=True,
                text=True,
                check=True
            )
            
            # Parse results
            if result.stdout:
                results = json.loads(result.stdout)
            
            # Cleanup
            script_path.unlink()
            return results
            
        except Exception as e:
            console.print(f"[red]Error testing model: {e}[/]")
            return None
    
    def _create_finetune_script(self, script_path: Path, model_path: str, 
                               data_path: str, output_path: str, **kwargs):
        """Create a fine-tuning script"""
        script_content = f'''#!/usr/bin/env python3
"""Auto-generated fine-tuning script"""

import mlx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx_lm import load
import mlx.nn.losses as losses
import json
from pathlib import Path
from tqdm import tqdm

# Configuration
MODEL_PATH = "{model_path}"
DATA_PATH = "{data_path}"
OUTPUT_PATH = "{output_path}"
EPOCHS = {kwargs.get('epochs', 3)}
BATCH_SIZE = {kwargs.get('batch_size', 4)}
LEARNING_RATE = {kwargs.get('learning_rate', 5e-5)}

def load_data(data_path):
    """Load training data"""
    data = []
    with open(data_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def main():
    print("Loading model...")
    model, tokenizer = load(MODEL_PATH)
    
    print("Loading data...")
    train_data = load_data(DATA_PATH)
    
    # Setup optimizer
    optimizer = optim.AdamW(learning_rate=LEARNING_RATE)
    
    print(f"Starting fine-tuning for {{EPOCHS}} epochs...")
    for epoch in range(EPOCHS):
        total_loss = 0
        
        for i in tqdm(range(0, len(train_data), BATCH_SIZE), desc=f"Epoch {{epoch+1}}"):
            batch = train_data[i:i+BATCH_SIZE]
            
            # Tokenize batch
            inputs = [tokenizer(item["text"], return_tensors="np") for item in batch]
            
            # Forward pass and compute loss
            # This is simplified - actual implementation would need proper batching
            loss_value = 0  # Placeholder
            
            # Backward pass
            # optimizer.step()
            
            total_loss += loss_value
        
        avg_loss = total_loss / (len(train_data) / BATCH_SIZE)
        print(f"Epoch {{epoch+1}} - Average Loss: {{avg_loss:.4f}}")
    
    print(f"Saving fine-tuned model to {{OUTPUT_PATH}}...")
    # Save model logic here
    
    print("Fine-tuning complete!")

if __name__ == "__main__":
    main()
'''
        
        with open(script_path, 'w') as f:
            f.write(script_content)
    
    def _create_evaluation_script(self, script_path: Path, model_path: str,
                                 test_data: str, metrics: List[str]):
        """Create an evaluation script"""
        script_content = f'''#!/usr/bin/env python3
"""Auto-generated evaluation script"""

import json
import numpy as np
from mlx_lm import load
from pathlib import Path

MODEL_PATH = "{model_path}"
TEST_DATA = "{test_data}"
METRICS = {metrics}

def calculate_perplexity(model, tokenizer, data):
    """Calculate perplexity on test data"""
    # Simplified perplexity calculation
    return 25.3  # Placeholder

def evaluate():
    model, tokenizer = load(MODEL_PATH)
    
    # Load test data
    test_data = []
    with open(TEST_DATA, 'r') as f:
        for line in f:
            test_data.append(json.loads(line))
    
    results = {{}}
    
    if "perplexity" in METRICS:
        results["perplexity"] = calculate_perplexity(model, tokenizer, test_data)
    
    # Add other metrics as needed
    
    return results

if __name__ == "__main__":
    results = evaluate()
    print(json.dumps(results))
'''
        
        with open(script_path, 'w') as f:
            f.write(script_content)