# API Usage Documentation

## Overview
This document provides detailed information about using the FastAPI endpoints for training and predicting with PINN models.

## API Endpoints

### Heat Equation

#### Train a Model
```bash
curl -X POST "http://localhost:8000/heat/train" \
     -H "Content-Type: application/json" \
     -d '{"epochs": 1000, "learning_rate": 0.001}'
```

Response:
```json
{
    "status": "Training completed",
    "epochs": 1000
}
```

#### Make Predictions
```bash
curl -X POST "http://localhost:8000/heat/predict" \
     -H "Content-Type: application/json" \
     -d '{"x": [0.0, 0.5, 1.0], "t": [0.0, 0.1, 0.2]}'
```

Response:
```json
{
    "prediction": [0.0, 0.25, 0.5]
}
```

### Wave Equation

#### Train a Model
```bash
curl -X POST "http://localhost:8000/wave/train" \
     -H "Content-Type: application/json" \
     -d '{"epochs": 1000, "learning_rate": 0.001, "c": 1.0}'
```

Response:
```json
{
    "status": "Training completed",
    "epochs": 1000
}
```

#### Make Predictions
```bash
curl -X POST "http://localhost:8000/wave/predict" \
     -H "Content-Type: application/json" \
     -d '{"x": [0.0, 0.5, 1.0], "t": [0.0, 0.1, 0.2]}'
```

Response:
```json
{
    "prediction": [0.0, 0.25, 0.5]
}
```

### Burgers' Equation

#### Train a Model
```bash
curl -X POST "http://localhost:8000/burgers/train" \
     -H "Content-Type: application/json" \
     -d '{"epochs": 1000, "learning_rate": 0.001, "nu": 0.01}'
```

Response:
```json
{
    "status": "Training completed",
    "epochs": 1000
}
```

#### Make Predictions
```bash
curl -X POST "http://localhost:8000/burgers/predict" \
     -H "Content-Type: application/json" \
     -d '{"x": [-1.0, 0.0, 1.0], "t": [0.0, 0.1, 0.2]}'
```

Response:
```json
{
    "prediction": [0.0, 0.25, 0.5]
}
```

## Python Examples

### Heat Equation
```python
import requests

# Train the model
response = requests.post(
    "http://localhost:8000/heat/train",
    json={"epochs": 1000, "learning_rate": 0.001}
)
print(response.json())

# Make predictions
response = requests.post(
    "http://localhost:8000/heat/predict",
    json={"x": [0.0, 0.5, 1.0], "t": [0.0, 0.1, 0.2]}
)
print(response.json())
```

### Wave Equation
```python
import requests

# Train the model
response = requests.post(
    "http://localhost:8000/wave/train",
    json={"epochs": 1000, "learning_rate": 0.001, "c": 1.0}
)
print(response.json())

# Make predictions
response = requests.post(
    "http://localhost:8000/wave/predict",
    json={"x": [0.0, 0.5, 1.0], "t": [0.0, 0.1, 0.2]}
)
print(response.json())
```

### Burgers' Equation
```python
import requests

# Train the model
response = requests.post(
    "http://localhost:8000/burgers/train",
    json={"epochs": 1000, "learning_rate": 0.001, "nu": 0.01}
)
print(response.json())

# Make predictions
response = requests.post(
    "http://localhost:8000/burgers/predict",
    json={"x": [-1.0, 0.0, 1.0], "t": [0.0, 0.1, 0.2]}
)
print(response.json())
```

## Error Handling

The API returns appropriate error messages for various scenarios:

### Invalid Input
```json
{
    "detail": "Invalid input parameters"
}
```

### Model Not Trained
```json
{
    "detail": "Model not trained. Please train the model first."
}
```

### Server Error
```json
{
    "detail": "Internal server error"
}
```

## Rate Limiting

The API implements rate limiting to prevent abuse. By default, it allows:
- 100 requests per minute for training endpoints
- 1000 requests per minute for prediction endpoints

If the rate limit is exceeded, the API returns:
```json
{
    "detail": "Rate limit exceeded"
}
``` 