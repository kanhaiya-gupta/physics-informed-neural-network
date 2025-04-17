from pydantic import BaseModel
from typing import List

class HeatPredictRequest(BaseModel):
    x: List[float]  # Spatial coordinates
    t: List[float]  # Time coordinates

class HeatPredictResponse(BaseModel):
    prediction: List[float]  # Predicted solution values