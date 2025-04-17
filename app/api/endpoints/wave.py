from fastapi import APIRouter, Depends
from pydantic import BaseModel
from src.training.wave_trainer import WaveTrainer
from src.models.wave_pinn import WavePINN
from app.schemas.wave import WavePredictRequest, WavePredictResponse
from src.utils.config_parser import load_config
from src.utils.logger import (
    log_api_request,
    log_api_response,
    log_api_error,
    log_training_start,
    log_training_progress,
    log_training_complete
)
import torch
import os
from fastapi import HTTPException
import time

router = APIRouter()

# Load default configuration
CONFIG_PATH = "configs/equations/wave.yaml"
default_config = load_config(CONFIG_PATH)

class TrainRequest(BaseModel):
    epochs: int = default_config.get("training", {}).get("epochs", 1000)
    learning_rate: float = default_config.get("training", {}).get("learning_rate", 0.001)

class TrainResponse(BaseModel):
    message: str
    final_loss: float
    training_time: float
    epochs: int
    config_used: dict

@router.post("/train", response_model=TrainResponse)
async def train_wave(request: TrainRequest):
    """Train the Wave PINN model."""
    endpoint = "/wave/train"
    log_api_request(endpoint, request.dict())
    
    try:
        # Load and update configuration
        config = load_config(CONFIG_PATH)
        config["training"]["epochs"] = request.epochs
        config["training"]["learning_rate"] = request.learning_rate
        
        # Log training start
        log_training_start("Wave", config)
        
        start_time = time.time()
        trainer = WaveTrainer()
        
        # Override trainer's logging to use our logger
        def log_progress(epoch: int, loss: float):
            log_training_progress("Wave", epoch, loss)
        trainer.log_progress = log_progress
        
        final_loss = trainer.train(epochs=request.epochs, lr=request.learning_rate)
        training_time = time.time() - start_time
        
        # Log training completion
        log_training_complete("Wave", final_loss, training_time)
        
        response = TrainResponse(
            message="Training completed successfully",
            final_loss=float(final_loss),
            training_time=training_time,
            epochs=request.epochs,
            config_used=config
        )
        
        log_api_response(endpoint, response.dict(), 200)
        return response
        
    except FileNotFoundError:
        error = f"Configuration file not found at {CONFIG_PATH}"
        log_api_error(endpoint, error, 500)
        raise HTTPException(status_code=500, detail=error)
    except Exception as e:
        log_api_error(endpoint, e, 500)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/predict", response_model=WavePredictResponse)
async def predict_wave(request: WavePredictRequest):
    """Make predictions using the trained Wave PINN model."""
    endpoint = "/wave/predict"
    log_api_request(endpoint, request.dict())
    
    model_path = os.path.join("results", "wave", "models", "model.pth")
    if not os.path.exists(model_path):
        error = "Model not found. Please train the model first."
        log_api_error(endpoint, error, 404)
        raise HTTPException(status_code=404, detail=error)

    try:
        model = WavePINN()
        model.load_state_dict(torch.load(model_path))
        model.eval()
        
        x = torch.tensor(request.x, dtype=torch.float32).view(-1, 1)
        t = torch.tensor(request.t, dtype=torch.float32).view(-1, 1)
        
        with torch.no_grad():
            prediction = model.predict(x, t)
        
        prediction_list = prediction.squeeze().tolist()
        if not isinstance(prediction_list, list):
            prediction_list = [prediction_list]
            
        response = WavePredictResponse(prediction=prediction_list)
        log_api_response(endpoint, response.dict(), 200)
        return response
        
    except Exception as e:
        log_api_error(endpoint, e, 500)
        raise HTTPException(status_code=500, detail=str(e))
