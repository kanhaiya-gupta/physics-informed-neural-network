from pydantic import BaseModel
from typing import List

class WavePredictRequest(BaseModel):
    x: float
    t: float

class WavePredictResponse(BaseModel):
    prediction: List[float]
