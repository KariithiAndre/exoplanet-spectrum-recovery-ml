# Contributing to Exoplanet Spectrum Recovery

Thank you for your interest in contributing to this project! This document provides guidelines for contributions.

## Getting Started

1. Fork the repository
2. Clone your fork locally
3. Create a new branch for your feature/fix
4. Make your changes
5. Submit a pull request

## Development Setup

### Prerequisites

- Python 3.10+
- Node.js 18+
- Git

### Backend Setup

```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install -e ".[dev]"  # Install dev dependencies
```

### Frontend Setup

```bash
cd frontend
npm install
```

### Running Tests

```bash
# Backend tests
cd backend
pytest

# Frontend tests
cd frontend
npm test
```

## Code Style

### Python

- Follow PEP 8 style guidelines
- Use type hints for function signatures
- Maximum line length: 88 characters (Black formatter)
- Run formatters before committing:

```bash
black .
isort .
flake8 .
```

### JavaScript/React

- Use ESLint and Prettier
- Prefer functional components with hooks
- Use meaningful component and variable names

```bash
npm run lint
```

## Pull Request Process

1. Ensure all tests pass
2. Update documentation if needed
3. Add a clear description of changes
4. Reference any related issues

## Reporting Issues

When reporting bugs, please include:

- Python/Node.js version
- Operating system
- Steps to reproduce
- Expected vs actual behavior
- Any error messages

## Scientific Contributions

For scientific contributions (new models, algorithms):

- Include references to relevant papers
- Provide validation against known results
- Document any assumptions or limitations
- Add appropriate unit tests

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
