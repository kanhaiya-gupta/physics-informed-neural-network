from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any
from pydantic import BaseModel
import torch
import os
from datetime import datetime

from src.utils.logger import (
    log_api_request,
    log_api_response,
    log_api_error,
    log_training_start,
    log_training_complete,
    get_equation_logger
)
from src.utils.config_parser import load_config

# Import trainers and models
from src.training.shm_trainer import SHMTrainer
from src.training.heat_trainer import HeatTrainer
from src.training.wave_trainer import WaveTrainer
from src.training.burgers_trainer import BurgersTrainer

router = APIRouter()

class BatchTrainRequest(BaseModel):
    equations: List[str]  # List of equations to train
    config: Dict[str, Any]  # Configuration for all equations

class BatchPredictRequest(BaseModel):
    equations: List[str]  # List of equations to predict
    points: Dict[str, List[float]]  # Points for each equation

class BatchResponse(BaseModel):
    status: str
    results: Dict[str, Any]

@router.post("/batch/train", response_model=BatchResponse)
async def batch_train(request: BatchTrainRequest):
    """Train multiple equations in batch."""
    log_api_request("/batch/train", request.dict())
    
    results = {}
    for equation in request.equations:
        try:
            logger = get_equation_logger(equation)
            log_training_start(equation, request.config)
            
            # Initialize appropriate trainer based on equation type
            if equation.lower() == "shm":
                trainer = SHMTrainer()
            elif equation.lower() == "heat":
                trainer = HeatTrainer()
            elif equation.lower() == "wave":
                trainer = WaveTrainer()
            elif equation.lower() == "burgers":
                trainer = BurgersTrainer()
            else:
                raise HTTPException(status_code=400, detail=f"Unsupported equation type: {equation}")
            
            # Train the model
            final_loss, training_time = trainer.train(
                epochs=request.config.epochs,
                lr=request.config.learning_rate,
                batch_size=request.config.batch_size
            )
            
            log_training_complete(equation, final_loss, training_time)
            results[equation] = {
                "status": "success",
                "final_loss": float(final_loss),
                "training_time": float(training_time)
            }
            
        except Exception as e:
            log_api_error(f"/batch/train/{equation}", e, 500)
            results[equation] = {
                "status": "error",
                "error": str(e)
            }
    
    response = BatchResponse(
        status="completed",
        results=results
    )
    log_api_response("/batch/train", response.dict(), 200)
    return response

@router.post("/batch/predict", response_model=BatchResponse)
async def batch_predict(request: BatchPredictRequest):
    """Make predictions for multiple equations in batch."""
    log_api_request("/batch/predict", request.dict())
    
    results = {}
    for equation in request.equations:
        try:
            # Check if model exists
            model_path = os.path.join("results", equation, "models", "model.pth")
            if not os.path.exists(model_path):
                raise HTTPException(
                    status_code=404,
                    detail=f"Model not found for {equation}. Please train the model first."
                )
            
            # Load model and make predictions
            if equation.lower() == "shm":
                from src.models.shm_pinn import SHMPINN
                model = SHMPINN()
                model.load_state_dict(torch.load(model_path))
                predictions = model(torch.tensor(request.points[equation], dtype=torch.float32))
            elif equation.lower() == "heat":
                from src.models.heat_pinn import HeatPINN
                model = HeatPINN()
                model.load_state_dict(torch.load(model_path))
                predictions = model(torch.tensor(request.points[equation], dtype=torch.float32))
            elif equation.lower() == "wave":
                from src.models.wave_pinn import WavePINN
                model = WavePINN()
                model.load_state_dict(torch.load(model_path))
                predictions = model(torch.tensor(request.points[equation], dtype=torch.float32))
            elif equation.lower() == "burgers":
                from src.models.burgers_pinn import BurgersPINN
                model = BurgersPINN()
                model.load_state_dict(torch.load(model_path))
                predictions = model(torch.tensor(request.points[equation], dtype=torch.float32))
            else:
                raise HTTPException(status_code=400, detail=f"Unsupported equation type: {equation}")
            
            results[equation] = {
                "status": "success",
                "predictions": predictions.detach().numpy().tolist()
            }
            
        except Exception as e:
            log_api_error(f"/batch/predict/{equation}", e, 500)
            results[equation] = {
                "status": "error",
                "error": str(e)
            }
    
    response = BatchResponse(
        status="completed",
        results=results
    )
    log_api_response("/batch/predict", response.dict(), 200)
    return response 