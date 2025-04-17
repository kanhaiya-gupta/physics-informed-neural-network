from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import torch
import os
from src.models.shm_pinn import SHMPINN
from src.equations.shm_equation import SHMEquation

router = APIRouter()

class SHMTrainRequest(BaseModel):
    epochs: int = 1000
    learning_rate: float = 0.001
    omega: float = 1.0

class SHMPredictRequest(BaseModel):
    t: list[float]

class SHMPredictResponse(BaseModel):
    prediction: list[float]

@router.post("/train")
async def train_shm(request: SHMTrainRequest):
    """Train the SHM PINN model."""
    try:
        from src.training.shm_trainer import SHMTrainer
        trainer = SHMTrainer(omega=request.omega)
        final_loss = trainer.train(epochs=request.epochs, lr=request.learning_rate)
        return {"message": "Training completed successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/predict")
async def predict_shm(request: SHMPredictRequest):
    """Make predictions with the trained SHM model."""
    try:
        # Check if model exists
        model_path = os.path.join("results", "shm", "models", "model.pth")
        if not os.path.exists(model_path):
            raise HTTPException(status_code=404, detail="Model not found. Please train the model first.")
        
        # Load model
        model = SHMPINN()
        model.load_state_dict(torch.load(model_path))
        model.eval()
        
        # Convert input to tensor
        t = torch.tensor(request.t, dtype=torch.float32).view(-1, 1)
        
        # Make prediction
        with torch.no_grad():
            prediction = model(t)
        
        # Convert prediction to list
        prediction_list = prediction.squeeze().tolist()
        if not isinstance(prediction_list, list):
            prediction_list = [prediction_list]
            
        return SHMPredictResponse(prediction=prediction_list)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 