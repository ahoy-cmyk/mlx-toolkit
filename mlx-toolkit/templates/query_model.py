#!/usr/bin/env python3
"""
Interactive Query Tool for MLX Models
Supports batch inference, streaming, and various generation strategies
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Generator
from dataclasses import dataclass

import mlx
import mlx.core as mx
import numpy as np
from rich.console import Console
from rich.prompt import Prompt
from rich.markdown import Markdown
from rich.panel import Panel
from rich.live import Live
from rich.table import Table


console = Console()


@dataclass
class GenerationConfig:
    """Generation configuration"""
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    do_sample: bool = True
    stream: bool = True
    seed: Optional[int] = None


class ModelServer:
    """Model server for inference"""
    
    def __init__(self, model_path: str, device: str = "gpu"):
        self.model_path = model_path
        self.device = device
        self.setup_model()
    
    def setup_model(self):
        """Load model and tokenizer"""
        console.print(f"[bold blue]Loading model from {self.model_path}...[/]")
        
        # Placeholder for actual MLX model loading
        # In real implementation:
        # from mlx_lm import load
        # self.model, self.tokenizer = load(self.model_path)
        
        # For now, use transformers tokenizer
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        console.print("[bold green]✓ Model loaded successfully![/]")
    
    def generate(self, prompt: str, config: GenerationConfig) -> str:
        """Generate response for a single prompt"""
        # Tokenize input - handle both TokenizerWrapper and regular tokenizers
        try:
            # First try with encode method (works with TokenizerWrapper)
            input_ids = self.tokenizer.encode(prompt)
            inputs = {"input_ids": [input_ids]}
        except Exception:
            # Fallback: try calling tokenizer as function
            inputs = self.tokenizer(prompt, return_tensors="np")
        
        input_ids = mx.array(inputs["input_ids"][0])
        
        # Generate (placeholder)
        # In real implementation:
        # from mlx_lm import generate
        # response = generate(
        #     self.model, self.tokenizer, prompt,
        #     max_tokens=config.max_tokens,
        #     temperature=config.temperature,
        #     top_p=config.top_p
        # )
        
        # Placeholder response
        response = f"Generated response for: '{prompt[:50]}...'"
        return response
    
    def generate_stream(self, prompt: str, config: GenerationConfig) -> Generator[str, None, None]:
        """Stream generation token by token"""
        # Placeholder streaming implementation
        response = f"This is a streaming response for the prompt: {prompt}"
        words = response.split()
        
        for word in words:
            yield word + " "
            time.sleep(0.05)  # Simulate generation delay
    
    def batch_generate(self, prompts: List[str], config: GenerationConfig) -> List[str]:
        """Generate responses for multiple prompts"""
        responses = []
        
        for prompt in prompts:
            response = self.generate(prompt, config)
            responses.append(response)
        
        return responses


class InteractiveChat:
    """Interactive chat interface"""
    
    def __init__(self, server: ModelServer, config: GenerationConfig):
        self.server = server
        self.config = config
        self.history = []
        self.system_prompt = None
    
    def set_system_prompt(self, prompt: str):
        """Set system prompt for the conversation"""
        self.system_prompt = prompt
        console.print(f"[yellow]System prompt set[/]")
    
    def format_prompt(self, user_input: str) -> str:
        """Format prompt with history and system prompt"""
        formatted = ""
        
        if self.system_prompt:
            formatted += f"System: {self.system_prompt}\n\n"
        
        # Add conversation history
        for entry in self.history[-5:]:  # Keep last 5 exchanges
            formatted += f"User: {entry['user']}\nAssistant: {entry['assistant']}\n\n"
        
        formatted += f"User: {user_input}\nAssistant:"
        return formatted
    
    def chat(self):
        """Start interactive chat"""
        console.print(Panel(
            "[bold cyan]Interactive Chat Mode[/]\n"
            "Commands:\n"
            "  /help - Show help\n"
            "  /clear - Clear history\n"
            "  /system <prompt> - Set system prompt\n"
            "  /save <file> - Save conversation\n"
            "  /load <file> - Load conversation\n"
            "  /config - Show generation config\n"
            "  /set <param> <value> - Update config\n"
            "  /quit - Exit",
            title="MLX Chat",
            border_style="cyan"
        ))
        
        while True:
            try:
                user_input = Prompt.ask("\n[bold blue]You[/]")
                
                if user_input.startswith("/"):
                    if not self.handle_command(user_input):
                        break
                    continue
                
                # Format prompt
                full_prompt = self.format_prompt(user_input)
                
                # Generate response
                console.print("\n[bold green]Assistant[/]: ", end="")
                
                if self.config.stream:
                    response = ""
                    with Live("", refresh_per_second=10, console=console) as live:
                        for token in self.server.generate_stream(full_prompt, self.config):
                            response += token
                            live.update(response)
                    console.print()  # New line after streaming
                else:
                    response = self.server.generate(full_prompt, self.config)
                    console.print(response)
                
                # Add to history
                self.history.append({
                    "user": user_input,
                    "assistant": response,
                    "timestamp": time.time()
                })
                
            except KeyboardInterrupt:
                console.print("\n[yellow]Interrupted. Type /quit to exit.[/]")
                continue
            except Exception as e:
                console.print(f"\n[red]Error: {e}[/]")
    
    def handle_command(self, command: str) -> bool:
        """Handle chat commands. Returns False to quit."""
        parts = command.split(maxsplit=1)
        cmd = parts[0].lower()
        
        if cmd == "/quit" or cmd == "/exit":
            return False
        
        elif cmd == "/help":
            self.show_help()
        
        elif cmd == "/clear":
            self.history.clear()
            console.print("[green]History cleared[/]")
        
        elif cmd == "/system" and len(parts) > 1:
            self.set_system_prompt(parts[1])
        
        elif cmd == "/save" and len(parts) > 1:
            self.save_conversation(parts[1])
        
        elif cmd == "/load" and len(parts) > 1:
            self.load_conversation(parts[1])
        
        elif cmd == "/config":
            self.show_config()
        
        elif cmd == "/set" and len(parts) > 1:
            self.update_config(parts[1])
        
        else:
            console.print("[red]Unknown command. Type /help for help.[/]")
        
        return True
    
    def show_help(self):
        """Show help information"""
        help_text = """
# Chat Commands

- `/help` - Show this help message
- `/clear` - Clear conversation history
- `/system <prompt>` - Set system prompt
- `/save <filename>` - Save conversation to file
- `/load <filename>` - Load conversation from file
- `/config` - Show current generation configuration
- `/set <param> <value>` - Update generation parameter
- `/quit` or `/exit` - Exit chat

# Generation Parameters

- `temperature` - Controls randomness (0.0-2.0)
- `top_p` - Nucleus sampling threshold (0.0-1.0)
- `top_k` - Top-k sampling (1-100)
- `max_tokens` - Maximum tokens to generate
- `stream` - Enable/disable streaming (true/false)
"""
        console.print(Markdown(help_text))
    
    def show_config(self):
        """Display current configuration"""
        table = Table(title="Generation Configuration")
        table.add_column("Parameter", style="cyan")
        table.add_column("Value", style="green")
        
        for field in self.config.__dataclass_fields__:
            value = getattr(self.config, field)
            table.add_row(field, str(value))
        
        console.print(table)
    
    def update_config(self, param_value: str):
        """Update configuration parameter"""
        try:
            param, value = param_value.split()
            
            if hasattr(self.config, param):
                # Convert value to appropriate type
                if param in ["max_tokens", "top_k"]:
                    value = int(value)
                elif param in ["temperature", "top_p", "repetition_penalty"]:
                    value = float(value)
                elif param in ["do_sample", "stream"]:
                    value = value.lower() == "true"
                
                setattr(self.config, param, value)
                console.print(f"[green]Updated {param} = {value}[/]")
            else:
                console.print(f"[red]Unknown parameter: {param}[/]")
        
        except Exception as e:
            console.print(f"[red]Error updating config: {e}[/]")
    
    def save_conversation(self, filename: str):
        """Save conversation to file"""
        try:
            with open(filename, 'w') as f:
                json.dump({
                    "system_prompt": self.system_prompt,
                    "history": self.history,
                    "config": self.config.__dict__
                }, f, indent=2)
            console.print(f"[green]Conversation saved to {filename}[/]")
        except Exception as e:
            console.print(f"[red]Error saving conversation: {e}[/]")
    
    def load_conversation(self, filename: str):
        """Load conversation from file"""
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
            
            self.system_prompt = data.get("system_prompt")
            self.history = data.get("history", [])
            
            # Update config
            for key, value in data.get("config", {}).items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
            
            console.print(f"[green]Conversation loaded from {filename}[/]")
        except Exception as e:
            console.print(f"[red]Error loading conversation: {e}[/]")


def main():
    parser = argparse.ArgumentParser(description="Query MLX models interactively or with prompts")
    parser.add_argument("model_path", help="Path to model")
    parser.add_argument("--prompt", "-p", help="Single prompt to generate response")
    parser.add_argument("--prompts-file", help="File with prompts (one per line)")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive chat mode")
    parser.add_argument("--max-tokens", type=int, default=512, help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature (0.0-2.0)")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p sampling")
    parser.add_argument("--top-k", type=int, default=50, help="Top-k sampling")
    parser.add_argument("--no-stream", action="store_true", help="Disable streaming")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--system", help="System prompt")
    parser.add_argument("--output", "-o", help="Output file for responses")
    
    args = parser.parse_args()
    
    # Create generation config
    config = GenerationConfig(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        stream=not args.no_stream,
        seed=args.seed
    )
    
    # Initialize model server
    server = ModelServer(args.model_path)
    
    # Handle different modes
    if args.interactive:
        # Interactive chat mode
        chat = InteractiveChat(server, config)
        if args.system:
            chat.set_system_prompt(args.system)
        chat.chat()
    
    elif args.prompts_file:
        # Batch mode
        with open(args.prompts_file, 'r') as f:
            prompts = [line.strip() for line in f if line.strip()]
        
        console.print(f"[blue]Processing {len(prompts)} prompts...[/]")
        responses = server.batch_generate(prompts, config)
        
        # Save or print responses
        if args.output:
            with open(args.output, 'w') as f:
                for prompt, response in zip(prompts, responses):
                    f.write(json.dumps({"prompt": prompt, "response": response}) + "\n")
            console.print(f"[green]Responses saved to {args.output}[/]")
        else:
            for prompt, response in zip(prompts, responses):
                console.print(Panel(
                    f"[bold]Prompt:[/] {prompt}\n\n[bold]Response:[/] {response}",
                    border_style="blue"
                ))
    
    elif args.prompt:
        # Single prompt mode
        response = server.generate(args.prompt, config)
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump({"prompt": args.prompt, "response": response}, f)
            console.print(f"[green]Response saved to {args.output}[/]")
        else:
            console.print(Panel(response, title="Response", border_style="green"))
    
    else:
        console.print("[red]Please provide --prompt, --prompts-file, or --interactive[/]")
        sys.exit(1)


if __name__ == "__main__":
    main()