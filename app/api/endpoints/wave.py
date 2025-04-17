from fastapi import APIRouter, Depends
from pydantic import BaseModel
from src.training.wave_trainer import WaveTrainer
from src.models.wave_pinn import WavePINN
from app.schemas.wave import WavePredictRequest, WavePredictResponse

router = APIRouter()

class TrainRequest(BaseModel):
    epochs: int = 1000
    learning_rate: float = 0.001
    c: float = 1.0  # Wave speed

@router.post("/train")
async def train_wave_model(request: TrainRequest):
    trainer = WaveTrainer(c=request.c)
    trainer.train(epochs=request.epochs, lr=request.learning_rate)
    return {"status": "Training completed", "epochs": request.epochs}

@router.post("/predict", response_model=WavePredictResponse)
async def predict_wave(request: WavePredictRequest):
    model = WavePINN()
    # Placeholder: Load trained model and predict
    prediction = model.predict(request.x, request.t)
    return WavePredictResponse(prediction=prediction.tolist())
