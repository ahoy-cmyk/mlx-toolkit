#!/usr/bin/env python3
"""MLX Toolkit Setup Script"""

from setuptools import setup, find_packages

setup(
    name="mlx-toolkit",
    version="0.1.0",
    description="All-in-one toolkit for MLX-based LLM development",
    author="Stephan Arrington",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "click>=8.0",
        "rich>=13.0",
        "pyyaml>=6.0",
        "psutil>=5.8.0",
    ],
    entry_points={
        "console_scripts": [
            "mlx-toolkit=mlx_toolkit.cli:main",
        ],
    },
)