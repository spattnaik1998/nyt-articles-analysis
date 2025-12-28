# Makefile for NYT Data Journalism Platform

.PHONY: help venv install clean run-api test lint format

# Variables
VENV_DIR = venv
PYTHON = $(VENV_DIR)/Scripts/python
PIP = $(VENV_DIR)/Scripts/pip
PYTEST = $(VENV_DIR)/Scripts/pytest
UVICORN = $(VENV_DIR)/Scripts/uvicorn

# Default target
help:
	@echo "NYT Data Journalism Platform - Available Commands"
	@echo "=================================================="
	@echo "make venv        - Create Python virtual environment"
	@echo "make install     - Install dependencies from requirements.txt"
	@echo "make run-api     - Start FastAPI development server"
	@echo "make test        - Run tests with coverage"
	@echo "make lint        - Run code linting (flake8)"
	@echo "make format      - Format code with black"
	@echo "make clean       - Remove virtual environment and cache files"
	@echo "make all         - Setup everything (venv + install)"

# Create virtual environment
venv:
	@echo "Creating virtual environment..."
	python -m venv $(VENV_DIR)
	@echo "Virtual environment created at $(VENV_DIR)/"
	@echo "Activate it with: $(VENV_DIR)\\Scripts\\activate (Windows) or source $(VENV_DIR)/bin/activate (Linux/Mac)"

# Install dependencies
install:
	@echo "Installing dependencies..."
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo "Dependencies installed successfully!"

# Run FastAPI server
run-api:
	@echo "Starting FastAPI development server..."
	$(UVICORN) src.api.main:app --reload --host 0.0.0.0 --port 8000

# Run tests with coverage
test:
	@echo "Running tests with coverage..."
	$(PYTEST) tests/ -v --cov=src --cov-report=term-missing --cov-report=html

# Run linting
lint:
	@echo "Running code linting..."
	$(PYTHON) -m flake8 src/ tests/ --max-line-length=120 --exclude=venv

# Format code
format:
	@echo "Formatting code with black..."
	$(PYTHON) -m black src/ tests/ --line-length=120

# Clean build artifacts and virtual environment
clean:
	@echo "Cleaning up..."
ifeq ($(OS),Windows_NT)
	@if exist $(VENV_DIR) rmdir /s /q $(VENV_DIR)
	@if exist __pycache__ rmdir /s /q __pycache__
	@if exist .pytest_cache rmdir /s /q .pytest_cache
	@if exist .coverage del /q .coverage
	@if exist htmlcov rmdir /s /q htmlcov
	@for /d /r . %%d in (__pycache__) do @if exist "%%d" rmdir /s /q "%%d"
	@del /s /q *.pyc 2>nul
else
	rm -rf $(VENV_DIR)
	rm -rf __pycache__
	rm -rf .pytest_cache
	rm -rf .coverage
	rm -rf htmlcov
	rm -rf *.egg-info
	rm -rf dist
	rm -rf build
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
endif
	@echo "Cleanup complete!"

# Setup everything
all: venv install
	@echo "Setup complete! Activate venv and run 'make run-api' to start."
