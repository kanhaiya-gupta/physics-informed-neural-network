from fastapi import FastAPI
from app.api.router import router

app = FastAPI(title="Physics-Informed Neural Network API")
app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)