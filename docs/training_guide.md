# ========================================================================
# docs/training_guide.md
# ========================================================================

# Training Guide

## Quick Start

### Basic Training
```bash
# Train with default settings
python src/train.py

# Train with GPU acceleration
python src/train.py --gpu

# Train improved model
python src/train.py --arch improved --epochs 15 --gpu
```

### Advanced Training
```bash
# Custom hyperparameters
python src/train.py \
    --arch improved \
    --epochs 20 \
    --batch_size 128 \
    --learning_rate 0.001 \
    --optimizer adamw \
    --scheduler onecycle \
    --gpu

# CPU-only training
python src/train.py --arch base --epochs 10 --batch_size 32
```

## Configuration Options

### Model Architecture
- `--arch base`: Standard 3-layer network (faster training)
- `--arch improved`: Enhanced 4-layer network (better accuracy)

### Training Parameters
- `--epochs`: Number of training epochs (default: 15)
- `--batch_size`: Batch size for training (default: 64)
- `--learning_rate`: Initial learning rate (default: 0.001)

### Optimization
- `--optimizer`: adam, adamw, or sgd (default: adamw)
- `--scheduler`: step, onecycle, or none (default: onecycle)

### Hardware
- `--gpu`: Enable GPU acceleration if available
- `--num_workers`: Number of data loading workers (default: 2)

## Training Process

### 1. Data Loading
- Automatic MNIST dataset download
- Train/test split: 60,000/10,000 images
- Data augmentation: rotation, translation
- Normalization: MNIST statistics (mean=0.1307, std=0.3081)

### 2. Model Initialization
- Weight initialization: Xavier (base) or Kaiming (improved)
- Architecture selection based on `--arch` parameter
- Model moved to appropriate device (CPU/GPU)

### 3. Training Loop
- Forward pass through the network
- Loss computation with CrossEntropyLoss
- Backward pass and gradient computation
- Parameter update using selected optimizer
- Learning rate scheduling

### 4. Validation
- Model evaluation on test set each epoch
- Metrics calculation: accuracy, loss
- Best model checkpoint saving

### 5. Final Evaluation
- Comprehensive test set evaluation
- Per-class accuracy analysis
- Confusion matrix generation
- Results visualization

## Monitoring and Logging

### Real-time Monitoring
- Progress bars with live metrics
- Training/validation loss tracking
- Accuracy progression monitoring

### Logging Output
```
Epoch [1/15] - Train Loss: 0.4523, Train Acc: 86.45%, Val Loss: 0.2341, Val Acc: 93.12%
Epoch [2/15] - Train Loss: 0.2156, Train Acc: 93.78%, Val Loss: 0.1456, Val Acc: 95.67%
...
```

### Saved Artifacts
- `best_model.pth`: Best performing model checkpoint
- `{arch}_final_model.pth`: Final model state
- Training curves visualization
- Confusion matrix plots

## Hyperparameter Tuning

### Learning Rate
```bash
# Conservative (stable but slower)
python src/train.py --learning_rate 0.0001

# Standard (recommended)
python src/train.py --learning_rate 0.001

# Aggressive (faster but less stable)
python src/train.py --learning_rate 0.01
```

### Batch Size
```bash
# Small batch (more stable gradients)
python src/train.py --batch_size 32

# Medium batch (balanced)
python src/train.py --batch_size 64

# Large batch (faster training)
python src/train.py --batch_size 128
```

### Optimization Strategy
```bash
# Conservative approach
python src/train.py --optimizer sgd --scheduler step

# Balanced approach
python src/train.py --optimizer adam --scheduler onecycle

# Advanced approach
python src/train.py --optimizer adamw --scheduler onecycle
```

## Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```bash
# Reduce batch size
python src/train.py --batch_size 32 --gpu

# Use CPU
python src/train.py --batch_size 64
```

**2. Slow Training**
```bash
# Increase batch size
python src/train.py --batch_size 128 --gpu

# Reduce number of workers if I/O bound
python src/train.py --num_workers 1
```

**3. Poor Convergence**
```bash
# Lower learning rate
python src/train.py --learning_rate 0.0001

# Change optimizer
python src/train.py --optimizer sgd
```

### Performance Optimization

**GPU Utilization**
- Use appropriate batch sizes for your GPU memory
- Enable mixed precision training for newer GPUs
- Monitor GPU utilization during training

**CPU Training**
- Reduce batch size to prevent memory issues
- Increase number of workers for data loading
- Consider distributed training for multiple CPUs

## Expected Results

### Base Model
- Training Accuracy: ~97.5%
- Validation Accuracy: ~97.8%
- Training Time: ~12 minutes (GPU)

### Improved Model
- Training Accuracy: ~99.0%
- Validation Accuracy: ~98.7%
- Training Time: ~15 minutes (GPU)

### Performance Targets
- **Minimum**: 95% accuracy (basic competency)
- **Good**: 97% accuracy (solid performance)
- **Excellent**: 98%+ accuracy (state-of-the-art for fully connected networks)
