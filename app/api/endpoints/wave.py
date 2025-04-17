from fastapi import APIRouter, Depends
from pydantic import BaseModel
from src.training.wave_trainer import WaveTrainer
from src.models.wave_pinn import WavePINN
from app.schemas.wave import WavePredictRequest, WavePredictResponse
import torch
import os

router = APIRouter()

class TrainRequest(BaseModel):
    epochs: int = 1000
    learning_rate: float = 0.001

@router.post("/train")
async def train_wave_model(request: TrainRequest):
    trainer = WaveTrainer()
    trainer.train(epochs=request.epochs, lr=request.learning_rate)
    return {"status": "Training completed", "epochs": request.epochs}

@router.post("/predict", response_model=WavePredictResponse)
async def predict_wave(request: WavePredictRequest):
    model = WavePINN()
    # Load trained model if it exists
    model_path = os.path.join("results", "wave", "models", "model.pth")
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        model.eval()
    else:
        return {"error": "Model not trained yet. Please train the model first."}
    
    # Convert inputs to tensors with correct shape
    x = torch.tensor(request.x, dtype=torch.float32).view(-1, 1)
    t = torch.tensor(request.t, dtype=torch.float32).view(-1, 1)
    
    # Get predictions
    with torch.no_grad():
        prediction = model.predict(x, t)
    
    # Convert prediction to list of floats
    prediction_list = prediction.squeeze().tolist()
    if not isinstance(prediction_list, list):
        prediction_list = [prediction_list]
    return WavePredictResponse(prediction=prediction_list)
