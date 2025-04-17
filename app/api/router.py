from fastapi import APIRouter
from .endpoints import shm, heat, wave, burgers

router = APIRouter()

router.include_router(shm.router, prefix="/shm", tags=["Simple Harmonic Motion"])
router.include_router(heat.router, prefix="/heat", tags=["Heat Equation"])
router.include_router(wave.router, prefix="/wave", tags=["Wave Equation"])
router.include_router(burgers.router, prefix="/burgers", tags=["Burgers' Equation"])