from fastapi import APIRouter, Depends
from pydantic import BaseModel
from src.training.heat_trainer import HeatTrainer
from src.models.heat_pinn import HeatPINN
from app.schemas.heat import HeatPredictRequest, HeatPredictResponse

router = APIRouter()

class TrainRequest(BaseModel):
    epochs: int = 1000
    learning_rate: float = 0.001

@router.post("/train")
async def train_heat_model(request: TrainRequest):
    trainer = HeatTrainer()
    trainer.train(epochs=request.epochs, lr=request.learning_rate)
    return {"status": "Training completed", "epochs": request.epochs}

@router.post("/predict", response_model=HeatPredictResponse)
async def predict_heat(request: HeatPredictRequest):
    model = HeatPINN()
    # Placeholder: Load trained model and predict
    prediction = model.predict(request.x, request.t)
    return HeatPredictResponse(prediction=prediction.tolist())