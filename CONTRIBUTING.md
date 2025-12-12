# Contributing to FBI UCR Crime Prediction Service

Thank you for your interest in contributing! This document outlines how to contribute effectively.

## Getting Started

1. Fork the repository
2. Clone your fork locally
3. Create a virtual environment and install dependencies:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -e ".[dev]"
   ```

## Development Workflow

### Making Changes

1. Create a feature branch from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. Make your changes following the code style guidelines below
3. Run tests to ensure nothing is broken
4. Commit with clear, descriptive messages

### Code Style

- Follow PEP 8 for Python code
- Use type hints where practical
- Keep functions focused and reasonably sized
- Write clear docstrings for public APIs

### Testing

Run the test suite before submitting:
```bash
pytest
pytest --cov=src/fbi_ucr  # with coverage
```

Add tests for new functionality and ensure existing tests pass.

## Submitting Changes

### Pull Requests

1. Push your branch to your fork
2. Open a pull request against `main`
3. Provide a clear description of:
   - What the change does
   - Why it's needed
   - Any relevant context or trade-offs

### Commit Messages

Use clear, descriptive commit messages:
- Start with a short summary (50 chars or less)
- Use imperative mood ("Add feature" not "Added feature")
- Reference issues when relevant

## Areas for Contribution

Here are some areas where contributions are welcome:

- **Additional offense types**: Extend support to more UCR crime categories
- **More states**: Add models for additional state-level predictions
- **Alternative models**: Experiment with other time series approaches (e.g., XGBoost, neural networks)
- **Improved explainability**: Enhance the prediction explanations
- **Documentation**: Improve docs, add examples, fix typos
- **Tests**: Increase test coverage

## Questions

If you have questions about contributing, feel free to open an issue for discussion.

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).
