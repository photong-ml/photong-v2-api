"""
Main API server.
"""

from os import getenv
from typing import Dict

from dotenv import load_dotenv
from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from util import predict_and_convert

load_dotenv()

app = FastAPI(
    title="Photong Prediction API",
    description="API for running inference on Photong models.",
    version="0.0.1",
    openapi_url="",
)

limiter = Limiter(key_func=get_remote_address, storage_uri=getenv("REDIS_ENDPOINT"))
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

origins = getenv("CORS_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root() -> Dict[str, str]:
    """Root."""
    return {"message": "Server is running"}


class PredictData(BaseModel):
    """Data for prediction."""

    img_data: str


@app.post("/predict")
@limiter.limit("1/minute")
async def predict(request: Request, data: PredictData) -> Response:
    """Predict."""
    try:
        return Response(predict_and_convert(data.img_data), media_type="audio/wav")
    except Exception as err:
        raise HTTPException(status_code=500, detail=str(err)) from err
