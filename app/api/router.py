from fastapi import APIRouter
from app.api.endpoints import shm, heat, wave, burgers, batch

api_router = APIRouter()

api_router.include_router(shm.router, prefix="/shm", tags=["Simple Harmonic Motion"])
api_router.include_router(heat.router, prefix="/heat", tags=["Heat Equation"])
api_router.include_router(wave.router, prefix="/wave", tags=["Wave Equation"])
api_router.include_router(burgers.router, prefix="/burgers", tags=["Burgers' Equation"])
api_router.include_router(batch.router, prefix="/batch", tags=["Batch Operations"])