from fastapi import APIRouter, Depends
from pydantic import BaseModel
from src.training.wave_trainer import WaveTrainer
from src.models.wave_pinn import WavePINN
from app.schemas.wave import WavePredictRequest, WavePredictResponse
import torch
import os
from fastapi import HTTPException

router = APIRouter()

class TrainRequest(BaseModel):
    epochs: int = 1000
    learning_rate: float = 0.001

@router.post("/train")
async def train_wave(request: TrainRequest):
    """Train the Wave PINN model."""
    try:
        trainer = WaveTrainer()
        final_loss = trainer.train(epochs=request.epochs, lr=request.learning_rate)
        return {"message": "Training completed successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/predict", response_model=WavePredictResponse)
async def predict_wave(request: WavePredictRequest):
    model_path = "results/wave/models/model.pth"
    if not os.path.exists(model_path):
        raise HTTPException(status_code=500, detail="Model has not been trained yet")

    # Initialize model and load weights
    model = WavePINN()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Convert inputs to tensors with correct shape
    x = torch.tensor([request.x], dtype=torch.float32).view(-1, 1)
    t = torch.tensor([request.t], dtype=torch.float32).view(-1, 1)

    # Make prediction
    with torch.no_grad():
        prediction = model(x, t)
        prediction_list = prediction.squeeze().tolist()
        if not isinstance(prediction_list, list):
            prediction_list = [prediction_list]
    
    return WavePredictResponse(prediction=prediction_list)
