from fastapi import FastAPI
import pickle
import pandas as pd
from pydantic import BaseModel, Field
from typing import Dict, Union, Optional

class PredictRequest(BaseModel):
    user_id: str = Field(..., description="Unique identifier for the user")
    data: Dict[str, Union[float, int, bool]] = Field(..., description="Features for prediction")
app = FastAPI()

# Load the model
with open('models/linear_regression.pkl', 'rb') as file:
    model = pickle.load(file)


@app.post("/test-predict/")
def test_predict(features: dict):
    print(features)
    return {"message": "Test prediction endpoint hit!", "features": features}

@app.post("/predict/")
def predict(request: PredictRequest):
    user_id = request.user_id
    features = request.data
    features_df = pd.DataFrame([features])
    prediction = model.predict(features_df)
    print(f"Prediction for user {user_id}: {prediction}")
    return {"prediction": prediction.tolist()}