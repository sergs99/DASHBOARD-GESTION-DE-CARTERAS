import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# Configuración básica de la aplicación
st.set_page_config(page_title="Análisis Financiero", layout="wide")

# Título de la aplicación
st.title("Herramientas de Análisis Financiero")

# Menú principal
menu = st.sidebar.selectbox(
    "Seleccione una categoría",
    ("Acciones", "Gestión de Carteras")
)

# Submenú en la categoría "Acciones"
if menu == "Acciones":
    submenu_acciones = st.sidebar.selectbox(
        "Seleccione un análisis para acciones",
        ("Análisis Técnico", "Análisis Fundamental", "Riesgo")
    )
    
    if submenu_acciones == "Análisis Técnico":
        st.subheader("Análisis Técnico")
        # Agrega aquí tu código para el análisis técnico
        st.write("Aquí puedes agregar el código para el análisis técnico.")
        
    elif submenu_acciones == "Análisis Fundamental":
        st.subheader("Análisis Fundamental")
        
        def calculate_financial_ratios(ticker):
            # Obtener los datos históricos y de la empresa
            stock = yf.Ticker(ticker)
            info = stock.info
            hist = stock.history(period="1y")
            
            # Asegúrate de que los datos están disponibles
            if hist.empty:
                raise ValueError("No data found for this ticker.")

           # Información financiera organizada
            fundamental_data = {
                'Nombre': info.get('shortName', 'N/A'),
                'Sector': info.get('sector', 'N/A'),
                'Industria': info.get('industry', 'N/A'),
                'Precio Actual': f"${info.get('currentPrice', 'N/A'):.2f}" if 'currentPrice' in info else 'N/A',
                'Ratios de Valoración': {
                    'Price Earnings Ratio': info.get('trailingPE', 'N/A'),
                    'Dividend Yield': f"{info.get('dividendYield', 'N/A')*100:.2f}%" if info.get('dividendYield') else 'N/A',
                    'Price to Book Value': info.get('priceToBook', 'N/A'),
                    'PEG Ratio (5yr expected)': info.get('pegRatio', 'N/A'),
                    'Price to Cash Flow Ratio': info.get('priceToCashflow', 'N/A'),
                    'EV/EBITDA': info.get('enterpriseToEbitda', 'N/A')
                },
                'Ratios de Rentabilidad': {
                    'Return on Equity': f"{info.get('returnOnEquity', 'N/A')*100:.2f}%" if info.get('returnOnEquity') else 'N/A',
                    'Return on Assets': f"{info.get('returnOnAssets', 'N/A')*100:.2f}%" if info.get('returnOnAssets') else 'N/A',
                    'Profit Margin': f"{info.get('profitMargins', 'N/A')*100:.2f}%" if info.get('profitMargins') else 'N/A',
                    'Operating Margin (ttm)': f"{info.get('operatingMargins', 'N/A')*100:.2f}%" if info.get('operatingMargins') else 'N/A',
                    'Payout Ratio': f"{info.get('payoutRatio', 'N/A')*100:.2f}%" if info.get('payoutRatio') else 'N/A'
                },
                'Ratios de Liquidez y Solvencia': {
                    'Current Ratio (mrq)': info.get('currentRatio', 'N/A'),
                    'Total Debt/Equity (mrq)': info.get('debtToEquity', 'N/A')
                },
                'Otras Métricas': {
                    'Volumen Actual': f"{info.get('volume', 'N/A'):,}" if 'volume' in info else 'N/A',
                    'Earnings Per Share (EPS)': info.get('trailingEps', 'N/A'),
                    'Capitalización de Mercado': f"${info.get('marketCap', 'N/A') / 1e9:.2f} B" if info.get('marketCap') else 'N/A',
                    'Beta': info.get('beta', 'N/A')
                }
            }

            for category, metrics in fundamental_data.items():
                st.write(f"**{category}:**")
                if isinstance(metrics, dict):
                    st.write(pd.DataFrame(list(metrics.items()), columns=['Métrica', 'Valor']).set_index('Métrica'))
                else:
                    st.write(metrics)

    elif submenu_acciones == "Riesgo":
        st.subheader("Análisis de Riesgo")
        # Agrega aquí tu código para el análisis de riesgo
        st.write("Aquí puedes agregar el código para el análisis de riesgo.")

# Submenú en la categoría "Gestión de Carteras"
elif menu == "Gestión de Carteras":
    submenu_carteras = st.sidebar.selectbox(
        "Seleccione una herramienta para gestión de carteras",
        ("Análisis de Carteras", "Optimización de Carteras")
    )
    
    if submenu_carteras == "Análisis de Carteras":
        st.subheader("Análisis de Carteras")
        # Agrega aquí tu código para el análisis de carteras
        st.write("Aquí puedes agregar el código para el análisis de carteras.")
        
    elif submenu_carteras == "Optimización de Carteras":
        st.subheader("Optimización de Carteras")
        # Agrega aquí tu código para la optimización de carteras
        st.write("Aquí puedes agregar el código para la optimización de carteras.")
