from fastapi import FastAPI
from app.api.router import api_router

app = FastAPI(title="PINN API", description="API for Physics-Informed Neural Networks")
app.include_router(api_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)