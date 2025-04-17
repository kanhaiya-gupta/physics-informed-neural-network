from pydantic import BaseModel
from typing import List

class BurgersPredictRequest(BaseModel):
    x: List[float]  # Spatial coordinates
    t: List[float]  # Time coordinates

class BurgersPredictResponse(BaseModel):
    prediction: List[float]  # Predicted solution values
