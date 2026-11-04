import os
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE, "model.pkl")
BUNDLE = joblib.load(MODEL_PATH)

if not isinstance(BUNDLE, dict) or not {"pipeline", "segment_map", "feature_order"}.issubset(BUNDLE):
    raise RuntimeError("model.pkl has an unexpected format. Use the supplied bundled model.pkl.")

PIPELINE = BUNDLE["pipeline"]
SEGMENT_MAP = BUNDLE["segment_map"]

app = FastAPI(title="Online Retail Customer Segmentation API", version="1.1.0")

class CustomerInput(BaseModel):
    recency: int = Field(..., ge=1, le=3650, description="Days since the latest purchase")
    frequency: int = Field(..., ge=1, le=1000, description="Number of orders")
    monetary: float = Field(..., ge=1, le=1_000_000, description="Total customer monetary value")

@app.get("/")
def root():
    return {"message": "Online Retail Customer Segmentation API", "docs": "/docs"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict")
def predict(customer: CustomerInput):
    try:
        X = np.log1p([[customer.recency, customer.frequency, customer.monetary]])
        cluster = int(PIPELINE.predict(X)[0])
        segment = SEGMENT_MAP.get(cluster, f"Cluster {cluster}")
        return {
            "segment": segment,
            "cluster": cluster,
            "input": customer.model_dump(),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
