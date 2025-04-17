from fastapi import APIRouter, Depends
from pydantic import BaseModel
from src.training.burgers_trainer import BurgersTrainer
from src.models.burgers_pinn import BurgersPINN
from app.schemas.burgers import BurgersPredictRequest, BurgersPredictResponse
import torch
import os

router = APIRouter()

class TrainRequest(BaseModel):
    epochs: int = 1000
    learning_rate: float = 0.001
    nu: float = 0.01  # Viscosity coefficient

@router.post("/train")
async def train_burgers_model(request: TrainRequest):
    trainer = BurgersTrainer(nu=request.nu)
    trainer.train(epochs=request.epochs, lr=request.learning_rate)
    return {"status": "Training completed", "epochs": request.epochs}

@router.post("/predict", response_model=BurgersPredictResponse)
async def predict_burgers(request: BurgersPredictRequest):
    model = BurgersPINN()
    # Load trained model if it exists
    model_path = os.path.join("results", "burgers", "models", "model.pth")
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
    return BurgersPredictResponse(prediction=prediction_list)
