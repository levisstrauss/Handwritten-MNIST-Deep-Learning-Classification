# ========================================================================
# MAKEFILE
# ========================================================================

.PHONY: install clean train predict test lint format help

# Default target
help:
	@echo "Available commands:"
	@echo "  install    - Install dependencies"
	@echo "  clean      - Clean generated files"
	@echo "  train      - Train the model"
	@echo "  predict    - Run prediction on sample image"
	@echo "  test       - Run tests"
	@echo "  lint       - Run code linting"
	@echo "  format     - Format code with black"
	@echo "  docker     - Build and run with Docker"

install:
	pip install -r requirements.txt
	pip install -e .

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf build/ dist/ *.egg-info/
	rm -rf logs/*
	rm -rf data/mnist*

train:
	python src/train.py --arch improved --epochs 15 --gpu

predict:
	python src/predict.py --image test_images/sample.png --model models/best_model.pth --confidence

test:
	python -m pytest tests/ -v

lint:
	flake8 src/ tests/
	pylint src/

format:
	black src/ tests/
	isort src/ tests/

docker:
	docker-compose up --build