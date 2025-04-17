from fastapi import APIRouter
from .endpoints import heat, wave, burgers

api_router = APIRouter()
api_router.include_router(heat.router, prefix="/heat", tags=["Heat Equation"])
api_router.include_router(wave.router, prefix="/wave", tags=["Wave Equation"])
api_router.include_router(burgers.router, prefix="/burgers", tags=["Burgers Equation"])