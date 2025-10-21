from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import pickle
import os

# ==============================
# 🚀 CONFIGURAÇÃO BÁSICA
# ==============================

app = FastAPI(
    title="FinPredict API",
    description="Sistema Inteligente de Previsão Financeira e Avaliação de Risco de Crédito",
    version="1.0.0",
)

# ==============================
# 📂 CAMINHOS DOS MODELOS
# ==============================

MODEL_PROFIT_PATH = os.path.join("models", "model_profit.pkl")
MODEL_CREDIT_PATH = os.path.join("models", "model_credit.pkl")


# ==============================
# 📊 MODELOS DE DADOS
# ==============================

class FinanceData(BaseModel):
    receita: float
    despesas: float
    investimentos: float


class CreditData(BaseModel):
    renda_mensal: float
    dividas: float
    score: float


# ==============================
# 🔰 ROTAS
# ==============================

@app.get("/")
def home():
    """Rota raiz da API"""
    return {"message": "🚀 FinPredict API rodando com sucesso!"}


@app.post("/predict/finance")
def predict_finance(data: FinanceData):
    """Previsão de lucro futuro"""
    try:
        if not os.path.exists(MODEL_PROFIT_PATH):
            raise FileNotFoundError("Modelo de previsão financeira não encontrado.")

        with open(MODEL_PROFIT_PATH, "rb") as f:
            model = pickle.load(f)

        X = np.array([[data.receita, data.despesas, data.investimentos]])
        y_pred = model.predict(X)

        return {"lucro_previsto": round(float(y_pred[0]), 2)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/credit")
def predict_credit(data: CreditData):
    """Cálculo de risco de crédito"""
    try:
        if not os.path.exists(MODEL_CREDIT_PATH):
            raise FileNotFoundError("Modelo de risco de crédito não encontrado.")

        with open(MODEL_CREDIT_PATH, "rb") as f:
            model = pickle.load(f)

        X = np.array([[data.renda_mensal, data.dividas, data.score]])
        y_pred = model.predict_proba(X)

        risco = ["Baixo", "Médio", "Alto"][int(np.argmax(y_pred))]
        return {
            "risco_credito": risco,
            "probabilidades": y_pred.tolist()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
