from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from src.training.heat_trainer import HeatTrainer
from src.models.heat_pinn import HeatPINN
from app.schemas.heat import HeatPredictRequest, HeatPredictResponse
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
import time

router = APIRouter()

# Load default configuration
CONFIG_PATH = "configs/equations/heat_equation.yaml"
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
async def train_heat(request: TrainRequest):
    """Train the Heat PINN model.
    
    Args:
        request: Training configuration including epochs and learning rate
        
    Returns:
        TrainResponse containing training metrics and status
        
    Raises:
        HTTPException: If training fails or config file is missing
    """
    endpoint = "/heat/train"
    log_api_request(endpoint, request.dict())
    
    try:
        # Load and update configuration
        config = load_config(CONFIG_PATH)
        config["training"]["epochs"] = request.epochs
        config["training"]["learning_rate"] = request.learning_rate
        
        # Log training start
        log_training_start("Heat", config)
        
        start_time = time.time()
        trainer = HeatTrainer()
        
        # Override trainer's logging to use our logger
        def log_progress(epoch: int, loss: float):
            log_training_progress("Heat", epoch, loss)
        trainer.log_progress = log_progress
        
        final_loss, training_time = trainer.train(epochs=request.epochs, lr=request.learning_rate)
        
        # Log training completion
        log_training_complete("Heat", final_loss, training_time)
        
        response = TrainResponse(
            message="Training completed successfully",
            final_loss=final_loss,
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

@router.post("/predict", response_model=HeatPredictResponse)
async def predict_heat(request: HeatPredictRequest):
    """Make predictions using the trained Heat PINN model.
    
    Args:
        request: Prediction inputs including spatial and temporal coordinates
        
    Returns:
        HeatPredictResponse containing model predictions
        
    Raises:
        HTTPException: If model is not found or prediction fails
    """
    endpoint = "/heat/predict"
    log_api_request(endpoint, request.dict())
    
    model_path = os.path.join("results", "heat", "models", "model.pth")
    if not os.path.exists(model_path):
        error = "Model not found. Please train the model first."
        log_api_error(endpoint, error, 404)
        raise HTTPException(status_code=404, detail=error)

    try:
        model = HeatPINN()
        model.load_state_dict(torch.load(model_path))
        model.eval()
        
        x = torch.tensor(request.x, dtype=torch.float32).view(-1, 1)
        t = torch.tensor(request.t, dtype=torch.float32).view(-1, 1)
        
        with torch.no_grad():
            prediction = model.predict(x, t)
        
        prediction_list = prediction.squeeze().tolist()
        if not isinstance(prediction_list, list):
            prediction_list = [prediction_list]
            
        response = HeatPredictResponse(prediction=prediction_list)
        log_api_response(endpoint, response.dict(), 200)
        return response
        
    except Exception as e:
        log_api_error(endpoint, e, 500)
        raise HTTPException(status_code=500, detail=str(e))