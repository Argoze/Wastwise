from fastapi import FastAPI
from pydantic import BaseModel
import joblib

# Carregar o modelo salvo
model = joblib.load("demand_forecast_model.pkl")

# Criar a aplicação FastAPI
app = FastAPI()

# Classe para entrada de dados
class ForecastInput(BaseModel):
    event: int
    temperature: float

# Endpoint para previsão
@app.post("/predict/")
def predict_demand(input_data: ForecastInput):
    # Preparar os dados de entrada
    features = [[input_data.event, input_data.temperature]]
    # Fazer a previsão
    prediction = model.predict(features)
    return {"predicted_quantity": prediction[0]}
