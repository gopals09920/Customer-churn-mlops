import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Customer Churn Prediction API")

# Absolute path resolution
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'xgb_churn_model.pkl')

class CustomerData(BaseModel):
    tenure: int
    MonthlyCharges: float
    TotalCharges: float
    SeniorCitizen: int = 0
    gender_Male: int = 0
    Partner_Yes: int = 0
    Dependents_Yes: int = 0
    Contract_One_year: int = 0
    Contract_Two_year: int = 0
    PaperlessBilling_Yes: int = 0
    PaymentMethod_Credit_card_automatic: int = 0
    PaymentMethod_Electronic_check: int = 0
    PaymentMethod_Mailed_check: int = 0

@app.get("/")
def home():
    return {"message": "Customer Churn Prediction API is Running!"}

@app.post("/predict")
def predict_churn(data: CustomerData):
    try:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file missing at: {MODEL_PATH}")
            
        model = joblib.load(MODEL_PATH)
        
        input_dict = data.dict()
        df_input = pd.DataFrame([input_dict])
        
        # Missing columns fix without warnings
        if hasattr(model, "feature_names_in_"):
            df_input = df_input.reindex(columns=model.feature_names_in_, fill_value=0)
            
        probability = float(model.predict_proba(df_input)[0][1])
        prediction = int(probability >= 0.5)
        
        return {
            "churn_prediction": "Yes" if prediction == 1 else "No",
            "churn_probability": round(probability, 4),
            "risk_level": "High" if probability > 0.6 else ("Medium" if probability > 0.3 else "Low")
        }
    except Exception as e:
        print(f"\n--- ERROR DURING PREDICTION --- \n{e}\n")
        raise HTTPException(status_code=500, detail=str(e))