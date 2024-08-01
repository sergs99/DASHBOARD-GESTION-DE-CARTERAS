import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# Función para obtener datos históricos de Yahoo Finance
def get_stock_data(ticker):
    stock = yf.Ticker(ticker)
    data = stock.history(period="1y")
    return data

# Función para calcular volatilidad histórica
def calculate_volatility(data):
    log_returns = np.log(data['Close'] / data['Close'].shift(1))
    volatility = np.std(log_returns.dropna()) * np.sqrt(252)
    return volatility

# Función para calcular indicadores financieros
def financial_ratios(data):
    latest = data.iloc[-1]
    ratios = {
        'Price to Earnings (P/E)': latest['Close'] / (latest['Close'] / 10),  # Placeholder for actual earnings data
        'Price to Book (P/B)': latest['Close'] / (latest['Close'] / 5),  # Placeholder for actual book value data
    }
    return ratios

# Título de la Aplicación y Estilo
st.set_page_config(page_title="Análisis de Empresas", layout="wide")
st.markdown("""
    <style>
    .main { background-color: #f4f4f4; }
    .sidebar .sidebar-content { background-color: #2E3B4E; }
    .sidebar .sidebar-content .element-container { color: white; }
    .sidebar .sidebar-content .block-container { color: white; }
    .sidebar .sidebar-content h1 { color: white; }
    .streamlit-expanderHeader { color: #006400; }
    </style>
""", unsafe_allow_html=True)

# Menú de Navegación
st.sidebar.title("Menú")
selection = st.sidebar.radio("Selecciona una opción", ["Inicio", "Datos Históricos", "Ratios Financieros", "Proyecciones"])

if selection == "Inicio":
    st.title("Análisis de Empresas")
    st.markdown("""
        **Bienvenido a la plataforma de análisis de empresas.**
        Esta aplicación permite analizar datos históricos de acciones, calcular ratios financieros,
        y realizar proyecciones simples sobre precios de acciones.
    """)

elif selection == "Datos Históricos":
    st.title("Datos Históricos de la Empresa")
    ticker = st.text_input("Ingrese el símbolo de la acción (ej. AAPL):").upper()
    
    if ticker:
        data = get_stock_data(ticker)
        if data.empty:
            st.write("No se encontraron datos para el ticker especificado.")
        else:
            st.subheader("Datos Históricos")
            st.line_chart(data['Close'])
            st.write(data.tail())

elif selection == "Ratios Financieros":
    st.title("Ratios Financieros")
    ticker = st.text_input("Ingrese el símbolo de la acción (ej. AAPL):").upper()
    
    if ticker:
        data = get_stock_data(ticker)
        if data.empty:
            st.write("No se encontraron datos para el ticker especificado.")
        else:
            ratios = financial_ratios(data)
            st.subheader("Ratios Financieros")
            st.write(pd.DataFrame(list(ratios.items()), columns=['Ratio', 'Valor']).set_index('Ratio'))

elif selection == "Proyecciones":
    st.title("Proyecciones de Precios")
    ticker = st.text_input("Ingrese el símbolo de la acción (ej. AAPL):").upper()
    
    if ticker:
        data = get_stock_data(ticker)
        if data.empty:
            st.write("No se encontraron datos para el ticker especificado.")
        else:
            st.subheader("Proyección basada en una Línea de Tendencia")
            x = np.arange(len(data))
            coefficients = np.polyfit(x, data['Close'], 1)
            trend = np.polyval(coefficients, x)
            plt.figure(figsize=(10, 6))
            plt.plot(data.index, data['Close'], label='Precio Real')
            plt.plot(data.index, trend, label='Línea de Tendencia', linestyle='--')
            plt.title('Proyección de Precios')
            plt.xlabel('Fecha')
            plt.ylabel('Precio de Cierre')
            plt.legend()
            st.pyplot(plt)
            
            # Cálculo y visualización de proyecciones
            st.subheader("Calcular P/E Ratio")
            earnings_per_share = st.number_input("Ingrese el beneficio por acción (EPS):", min_value=0.0, format="%.2f")
            if earnings_per_share > 0:
                pe_ratio = data['Close'].iloc[-1] / earnings_per_share
                st.write(f"El P/E Ratio calculado es: {pe_ratio:.2f}")
