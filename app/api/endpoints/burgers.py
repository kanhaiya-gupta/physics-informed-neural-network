from fastapi import APIRouter, Depends
from pydantic import BaseModel
from src.training.burgers_trainer import BurgersTrainer
from src.models.burgers_pinn import BurgersPINN
from app.schemas.burgers import BurgersPredictRequest, BurgersPredictResponse

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
    # Placeholder: Load trained model and predict
    prediction = model.predict(request.x, request.t)
    return BurgersPredictResponse(prediction=prediction.tolist())
