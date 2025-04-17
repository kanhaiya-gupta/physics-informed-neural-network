from fastapi import APIRouter, Depends
from pydantic import BaseModel
from src.training.heat_trainer import HeatTrainer
from src.models.heat_pinn import HeatPINN
from app.schemas.heat import HeatPredictRequest, HeatPredictResponse
import torch
import os
from fastapi import HTTPException

router = APIRouter()

class TrainRequest(BaseModel):
    epochs: int = 1000
    learning_rate: float = 0.001

@router.post("/train")
async def train_heat(request: TrainRequest):
    """Train the Heat PINN model."""
    try:
        trainer = HeatTrainer()
        final_loss = trainer.train(epochs=request.epochs, lr=request.learning_rate)
        return {"message": "Training completed successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/predict", response_model=HeatPredictResponse)
async def predict_heat(request: HeatPredictRequest):
    model = HeatPINN()
    # Load trained model if it exists
    model_path = os.path.join("results", "heat", "models", "model.pth")
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
    return HeatPredictResponse(prediction=prediction_list)