# Makefile for MATS abstention direction project

.PHONY: fix lint test test-unit test-integration test-analysis test-examples check ci coverage clean help

# Default target
help:
	@echo "Available targets:"
	@echo "  fix            - Format code and auto-fix lint with Black + Ruff"
	@echo "  lint           - Run Ruff linter"
	@echo "  test           - Run all tests"
	@echo "  test-unit      - Run unit tests only"
	@echo "  test-integration - Run integration tests only"  
	@echo "  test-analysis  - Run analysis tests only"
	@echo "  test-examples  - Run example tests only"
	@echo "  test-fast      - Run tests excluding slow ones"
	@echo "  check          - Run lint and test"
	@echo "  ci             - Run fix, lint, and test (CI pipeline)"
	@echo "  coverage       - Generate test coverage report (HTML)"
	@echo "  clean          - Clean test artifacts and caches"
	@echo "  help           - Show this help message"

# Code formatting and linting
fix:
	@echo "Formatting code with Black..."
	@python -m black . || echo "Black not installed, skipping formatting"
	@echo "Auto-fixing lint issues with Ruff..."
	@python -m ruff check --fix . || echo "Some lint issues require manual fixing"

lint:
	@echo "Running Ruff linter..."
	@python -m ruff check . || echo "Ruff not installed, skipping lint check"

# Run all tests
test:
	pytest

# Run tests by category
test-unit:
	pytest tests/unit/

test-integration:
	pytest tests/integration/

test-analysis:
	pytest tests/analysis/

test-examples:
	pytest tests/examples/

# Run fast tests only (exclude slow marked tests)
test-fast:
	pytest -m "not slow"

# Quality checks
check: lint test

# CI pipeline
ci: fix lint test

# Test coverage
coverage:
	@echo "Generating test coverage report..."
	@pytest --cov=shrugger --cov-report=html --cov-report=term || echo "Coverage tools not installed"

# Clean test artifacts and caches
clean:
	@echo "Cleaning up test artifacts and caches..."
	@find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	@find . -name "*.pyc" -delete 2>/dev/null || true
	@find . -name ".coverage" -delete 2>/dev/null || true
	@rm -rf htmlcov/ 2>/dev/null || true
	@rm -rf .ruff_cache/ 2>/dev/null || true
	@echo "Cleanup complete"