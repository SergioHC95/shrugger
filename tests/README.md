# Test Suite Organization

This directory contains all tests for the shrugger project, organized by test type and functionality.

## Directory Structure

```
tests/
├── unit/                   # Unit tests for individual functions/classes
│   └── test_metrics.py    # Tests for CA score, HEDGE score calculations
├── integration/           # Integration tests for full pipelines  
│   └── test_full_pipeline.py  # End-to-end pipeline testing
├── analysis/              # Tests for Fisher LDA analysis functionality
│   ├── test_fisher_lda.py       # Comprehensive Fisher LDA tests
│   ├── test_analysis_direct.py  # Direct module testing
│   └── test_analysis_standalone.py  # Standalone functionality tests
└── examples/              # Tests for example code
    └── test_likert_descriptions.py  # Example code validation
```

## Running Tests

### Run All Tests
```bash
pytest
```

### Run Tests by Category
```bash
# Unit tests only
pytest tests/unit/

# Integration tests only  
pytest tests/integration/

# Analysis tests only
pytest tests/analysis/

# Example tests only
pytest tests/examples/
```

### Run Tests by Marker
```bash
# Run only unit tests (if marked)
pytest -m unit

# Run only integration tests (if marked)
pytest -m integration

# Run only analysis tests (if marked)
pytest -m analysis

# Skip slow tests
pytest -m "not slow"
```

### Run Specific Test Files
```bash
# Test Fisher LDA functionality
pytest tests/analysis/test_fisher_lda.py

# Test metrics calculations
pytest tests/unit/test_metrics.py

# Test full pipeline
pytest tests/integration/test_full_pipeline.py
```

## Test Categories

### Unit Tests (`tests/unit/`)
- Test individual functions and classes in isolation
- Fast execution, no external dependencies
- Focus on mathematical correctness and edge cases

### Integration Tests (`tests/integration/`)
- Test complete workflows and pipelines
- May involve file I/O, data loading, and multi-step processes
- Ensure components work together correctly

### Analysis Tests (`tests/analysis/`)
- Specific to Fisher LDA analysis functionality
- Test the core refactored analysis modules
- Include both unit and integration aspects of analysis code

### Example Tests (`tests/examples/`)
- Validate example code and demonstrations
- Ensure examples remain functional as codebase evolves
- Test tutorial and documentation code

## Test Markers

Tests can be marked with pytest markers for selective execution:

- `@pytest.mark.unit`: Unit tests
- `@pytest.mark.integration`: Integration tests  
- `@pytest.mark.analysis`: Analysis-specific tests
- `@pytest.mark.examples`: Example code tests
- `@pytest.mark.slow`: Tests that take significant time

## Adding New Tests

1. **Choose the right directory** based on test type
2. **Follow naming convention**: `test_*.py` for files, `test_*` for functions
3. **Add appropriate markers** if using pytest markers
4. **Include docstrings** explaining what the test validates
5. **Use descriptive test names** that explain the scenario being tested

## Configuration

Test configuration is managed in `pytest.ini` at the project root, including:
- Test discovery paths
- Warning filters
- Default options
- Marker definitions
