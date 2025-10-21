import sys
import os

# Corrige o path para conseguir importar src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import streamlit as st
import pandas as pd
from src.utils.data_loader import load_data
from src.gemini.gemini_api import generate_insight

st.title("FinPredict Dashboard")

# Carregar dados
data = load_data('data/dataset.csv')
st.subheader("Dados de Entrada")
st.write(data)

# Mostrar gráficos simples
st.subheader("Gráfico de Previsão")
st.line_chart(data[['feature1', 'feature2', 'target']])

# Gerar insight via Gemini
if st.button("Gerar Insight"):
    insight = generate_insight("Analise os resultados da previsão financeira e risco de crédito.")
    st.subheader("Insight Gemini")
    st.write(insight)
