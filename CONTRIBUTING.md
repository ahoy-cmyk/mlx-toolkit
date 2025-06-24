# Contributing to MLX Toolkit

Thank you for your interest in contributing to MLX Toolkit! We welcome contributions from the community.

## How to Contribute

### Reporting Issues

- Use the GitHub issue tracker to report bugs
- Describe the issue clearly and provide steps to reproduce
- Include your system information (macOS version, Python version, etc.)

### Submitting Pull Requests

1. Fork the repository
2. Create a new branch for your feature (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests to ensure nothing is broken
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to your branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

### Development Setup

```bash
# Clone your fork
git clone https://github.com/ahoy-cmyk/mlx-toolkit.git
cd mlx-toolkit

# Install in development mode
cd mlx-toolkit
./install.sh
source venv/bin/activate
pip install -e .

# Install dev dependencies
pip install pytest pytest-cov black flake8
```

### Code Style

- Follow PEP 8
- Use meaningful variable and function names
- Add docstrings to all functions and classes
- Run `black` for code formatting

### Testing

- Add tests for new features
- Ensure all tests pass before submitting PR
- Run tests with: `pytest tests/`

## Areas for Contribution

- Adding support for more models
- Improving fine-tuning algorithms
- Adding new evaluation metrics
- Documentation improvements
- Performance optimizations
- Bug fixes

## Questions?

Feel free to open an issue for any questions about contributing!