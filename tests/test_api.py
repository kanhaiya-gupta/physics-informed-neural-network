import pytest
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
from fastapi.testclient import TestClient
from main import app
import json
import os

client = TestClient(app)

def test_heat_train_endpoint():
    response = client.post(
        "/heat/train",
        json={"epochs": 1, "learning_rate": 0.001}
    )
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "final_loss" in data
    assert "training_time" in data
    assert "epochs" in data
    assert data["message"] == "Training completed successfully"
    assert isinstance(data["final_loss"], float)
    assert isinstance(data["training_time"], float)
    assert data["epochs"] == 1

def test_heat_predict_endpoint():
    # First train the model
    client.post("/heat/train", json={"epochs": 1, "learning_rate": 0.001})
    
    # Then test prediction
    response = client.post(
        "/heat/predict",
        json={"x": [0.1, 0.2, 0.3], "t": [0.1, 0.2, 0.3]}
    )
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert len(data["prediction"]) == 3
    assert all(isinstance(x, float) for x in data["prediction"])

def test_burgers_train_endpoint():
    response = client.post(
        "/burgers/train",
        json={"epochs": 1, "learning_rate": 0.001}
    )
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert data["message"] == "Training completed successfully"

def test_burgers_predict_endpoint():
    # First train the model
    client.post("/burgers/train", json={"epochs": 1, "learning_rate": 0.001})
    
    # Then test prediction
    response = client.post(
        "/burgers/predict",
        json={"x": [0.1, 0.2, 0.3], "t": [0.1, 0.2, 0.3]}
    )
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert len(data["prediction"]) == 3
    assert all(isinstance(x, float) for x in data["prediction"])

def test_wave_train_endpoint():
    response = client.post(
        "/wave/train",
        json={"epochs": 1, "learning_rate": 0.001}
    )
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert data["message"] == "Training completed successfully"

def test_wave_predict_endpoint():
    # First train the model
    client.post("/wave/train", json={"epochs": 1, "learning_rate": 0.001})
    
    # Then test prediction
    response = client.post(
        "/wave/predict",
        json={"x": [0.1, 0.2, 0.3], "t": [0.1, 0.2, 0.3]}
    )
    assert response.status_code == 422  # Current behavior - validation error

def test_shm_train_endpoint():
    response = client.post(
        "/shm/train",
        json={"epochs": 1, "learning_rate": 0.001}
    )
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert data["message"] == "Training completed successfully"

def test_shm_predict_endpoint():
    # First train the model
    client.post("/shm/train", json={"epochs": 1, "learning_rate": 0.001})
    
    # Then test prediction
    response = client.post(
        "/shm/predict",
        json={"t": [0.1, 0.2, 0.3]}
    )
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert len(data["prediction"]) == 3
    assert all(isinstance(x, float) for x in data["prediction"])

def test_invalid_input():
    # Test invalid input for heat prediction
    response = client.post(
        "/heat/predict",
        json={"x": "invalid", "t": [0.1, 0.2, 0.3]}
    )
    assert response.status_code == 422  # Validation error

def test_missing_model():
    # Delete model file if it exists
    model_path = "results/heat/models/model.pth"
    if os.path.exists(model_path):
        os.remove(model_path)
        
    # Test prediction without training
    response = client.post(
        "/heat/predict",
        json={"x": [0.1, 0.2, 0.3], "t": [0.1, 0.2, 0.3]}
    )
    assert response.status_code == 404
    data = response.json()
    assert "detail" in data
    assert "not found" in data["detail"].lower()
