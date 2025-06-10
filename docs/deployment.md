# ========================================================================
# docs/deployment.md
# ========================================================================

# Deployment Guide

## Overview

This guide covers various deployment strategies for the MNIST digit classification model, from development environments to production systems.

## Local Deployment

### Development Setup
```bash
# Clone repository
git clone https://github.com/[username]/MNIST-Deep-Learning-Classification.git
cd MNIST-Deep-Learning-Classification

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

### Model Training
```bash
# Train model locally
python src/train.py --arch improved --epochs 15 --gpu

# Verify model performance
python src/predict.py --image test_images/sample.png --model models/best_model.pth
```

## Docker Deployment

### Build and Run
```bash
# Build Docker image
docker build -t mnist-classifier .

# Run training container
docker run --gpus all -v $(pwd)/models:/app/models mnist-classifier \
    python src/train.py --arch improved --epochs 15 --gpu

# Run prediction container
docker run -v $(pwd)/models:/app/models -v $(pwd)/test_images:/app/test_images \
    mnist-classifier python src/predict.py \
    --image test_images/sample.png --model models/best_model.pth
```

### Docker Compose
```bash
# Start services
docker-compose up --build

# Run specific service
docker-compose run mnist-trainer
docker-compose run mnist-predictor
```

## Cloud Deployment

### AWS Deployment

**1. ECR (Elastic Container Registry)**
```bash
# Create ECR repository
aws ecr create-repository --repository-name mnist-classifier

# Get login token
aws ecr get-login-password --region us-east-1 | \
    docker login --username AWS --password-stdin [account-id].dkr.ecr.us-east-1.amazonaws.com

# Tag and push image
docker tag mnist-classifier:latest [account-id].dkr.ecr.us-east-1.amazonaws.com/mnist-classifier:latest
docker push [account-id].dkr.ecr.us-east-1.amazonaws.com/mnist-classifier:latest
```

**2. ECS (Elastic Container Service)**
```bash
# Create task definition
aws ecs register-task-definition --cli-input-json file://ecs-task-definition.json

# Create service
aws ecs create-service --cluster mnist-cluster --service-name mnist-service \
    --task-definition mnist-classifier --desired-count 1
```

**3. Lambda Deployment**
```python
# lambda_function.py
import json
import base64
import torch
from PIL import Image
import io
from src.models import ImprovedMNISTClassifier
from src.utils import load_model

def lambda_handler(event, context):
    # Decode base64 image
    image_data = base64.b64decode(event['image'])
    image = Image.open(io.BytesIO(image_data))
    
    # Load model
    model = ImprovedMNISTClassifier()
    load_model(model, '/opt/models/best_model.pth', torch.device('cpu'))
    
    # Make prediction
    # ... prediction code ...
    
    return {
        'statusCode': 200,
        'body': json.dumps({'prediction': prediction})
    }
```

### Google Cloud Platform

**1. Cloud Run**
```bash
# Build and push to Container Registry
gcloud builds submit --tag gcr.io/[PROJECT-ID]/mnist-classifier

# Deploy to Cloud Run
gcloud run deploy --image gcr.io/[PROJECT-ID]/mnist-classifier --platform managed
```

**2. AI Platform**
```bash
# Package model
tar -czf model.tar.gz models/

# Upload to Cloud Storage
gsutil cp model.tar.gz gs://[BUCKET]/models/

# Create model version
gcloud ai-platform versions create v1 --model mnist_classifier \
    --origin gs://[BUCKET]/models/ --runtime-version 2.1 --python-version 3.7
```

### Azure Deployment

**1. Container Instances**
```bash
# Create resource group
az group create --name mnist-rg --location eastus

# Deploy container
az container create --resource-group mnist-rg --name mnist-classifier \
    --image [registry]/mnist-classifier:latest --cpu 2 --memory 4
```

**2. Machine Learning Service**
```python
# deploy.py
from azureml.core import Workspace, Model
from azureml.core.webservice import AciWebservice, Webservice

# Register model
model = Model.register(workspace=ws, model_path="models/best_model.pth", 
                      model_name="mnist-classifier")

# Deploy to ACI
aci_config = AciWebservice.deploy_configuration(cpu_cores=1, memory_gb=1)
service = Model.deploy(workspace=ws, name="mnist-service", models=[model], 
                      deployment_config=aci_config)
```

## Edge Deployment

### Mobile Deployment (PyTorch Mobile)
```python
# optimize_mobile.py
import torch
from src.models import ImprovedMNISTClassifier

# Load trained model
model = ImprovedMNISTClassifier()
checkpoint = torch.load('models/best_model.pth', map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Optimize for mobile
example_input = torch.randn(1, 1, 28, 28)
traced_model = torch.jit.trace(model, example_input)
traced_model_optimized = torch.jit.optimize_for_inference(traced_model)

# Save optimized model
traced_model_optimized.save('models/mnist_mobile.pt')
```

### Raspberry Pi Deployment
```bash
# Install PyTorch for ARM
pip install torch torchvision -f https://download.pytorch.org/whl/torch_stable.html

# Copy optimized model
scp models/mnist_mobile.pt pi@raspberry-pi:/home/pi/models/

# Run inference
python src/predict.py --image captured_digit.png --model models/mnist_mobile.pt
```

## Production Considerations

### API Server Setup
```python
# api_server.py
from flask import Flask, request, jsonify
import torch
from PIL import Image
import io
import base64

app = Flask(__name__)

# Load model once at startup
model = load_model_from_checkpoint('models/best_model.pth')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get image from request
        image_data = request.json['image']
        image = decode_base64_image(image_data)
        
        # Make prediction
        prediction = predict_digit(model, image)
        
        return jsonify({
            'prediction': int(prediction),
            'confidence': float(confidence),
            'status': 'success'
        })
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
```

### Load Balancing
```yaml
# kubernetes/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mnist-classifier
spec:
  replicas: 3
  selector:
    matchLabels:
      app: mnist-classifier
  template:
    metadata:
      labels:
        app: mnist-classifier
    spec:
      containers:
      - name: mnist-classifier
        image: mnist-classifier:latest
        ports:
        - containerPort: 8080
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
```

### Monitoring and Logging
```python
# monitoring.py
import logging
import time
from prometheus_client import Counter, Histogram, start_http_server

# Metrics
prediction_count = Counter('predictions_total', 'Total predictions made')
prediction_latency = Histogram('prediction_duration_seconds', 'Prediction latency')

def monitored_prediction(model, image):
    start_time = time.time()
    
    try:
        prediction = predict_digit(model, image)
        prediction_count.inc()
        
        latency = time.time() - start_time
        prediction_latency.observe(latency)
        
        logging.info(f"Prediction: {prediction}, Latency: {latency:.3f}s")
        return prediction
        
    except Exception as e:
        logging.error(f"Prediction failed: {str(e)}")
        raise

# Start metrics server
start_http_server(8000)
```

## Performance Optimization

### Model Optimization
```python
# Quantization for CPU deployment
import torch.quantization as quantization

# Post-training quantization
model_fp32 = load_model()
model_fp32.eval()
model_int8 = quantization.quantize_dynamic(model_fp32, {torch.nn.Linear}, dtype=torch.qint8)

# Save quantized model
torch.save(model_int8.state_dict(), 'models/mnist_quantized.pth')
```

### Batch Inference
```python
# batch_inference.py
def batch_predict(model, images, batch_size=32):
    """Optimized batch prediction."""
    predictions = []
    
    for i in range(0, len(images), batch_size):
        batch = images[i:i+batch_size]
        batch_tensor = torch.stack([preprocess_image(img) for img in batch])
        
        with torch.no_grad():
            outputs = model(batch_tensor)
            batch_predictions = torch.argmax(outputs, dim=1)
            predictions.extend(batch_predictions.cpu().numpy())
    
    return predictions
```

## Security Considerations

### Input Validation
```python
def validate_input(image_data):
    """Validate input image."""
    if len(image_data) > 10 * 1024 * 1024:  # 10MB limit
        raise ValueError("Image too large")
    
    try:
        image = Image.open(io.BytesIO(image_data))
        if image.size != (28, 28) and max(image.size) > 1000:
            raise ValueError("Invalid image dimensions")
    except Exception:
        raise ValueError("Invalid image format")
```

### Rate Limiting
```python
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

limiter = Limiter(
    app,
    key_func=get_remote_address,
    default_limits=["1000 per hour", "100 per minute"]
)

@app.route('/predict', methods=['POST'])
@limiter.limit("10 per minute")
def predict():
    # ... prediction code ...
```