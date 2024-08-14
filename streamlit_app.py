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
        
    if submenu_acciones == "Análisis Fundamental":
        st.subheader("Análisis Fundamental")
def calculate_financial_ratios(ticker):
    # Obtener los datos históricos y de la empresa
    stock = yf.Ticker(ticker)
    info = stock.info
    hist = stock.history(period="1y")
    
    # Asegúrate de que los datos están disponibles
    if hist.empty:
        raise ValueError("No data found for this ticker.")

    # Ratios de Valoración
    current_price = info.get('currentPrice', 1)
    eps = info.get('trailingEps', None)  # Earnings Per Share
    trailing_pe = info.get('trailingPE', None)  # Price Earnings Ratio
    dividend_yield = info.get('dividendYield', None)  # Dividend Yield
    price_to_book_value = info.get('priceToBook', None)  # Price to Book Value
    peg_ratio = info.get('pegRatio', None)  # PEG Ratio
    cash_flow = stock.cashflow.loc['Total Cash From Operating Activities'].iloc[0] if 'Total Cash From Operating Activities' in stock.cashflow.index else None
    shares_outstanding = info.get('sharesOutstanding', 1)  # Shares Outstanding
    price_to_cash_flow = current_price / (cash_flow / shares_outstanding) if cash_flow else None
    ev_to_ebitda = info.get('enterpriseToEbitda', None)  # EV/EBITDA

    # Ratios de Rentabilidad
    return_on_equity = info.get('returnOnEquity', None) * 100 if info.get('returnOnEquity', None) else None
    profit_margin = info.get('profitMargins', None) * 100 if info.get('profitMargins', None) else None
    operating_margin = info.get('operatingMargins', None) * 100 if info.get('operatingMargins', None) else None
    payout_ratio = info.get('payoutRatio', None) * 100 if info.get('payoutRatio', None) else None

    # Ratios de Liquidez y Solvencia
    balance_sheet = stock.balance_sheet
    current_assets = balance_sheet.loc['Total Current Assets'].iloc[0] if 'Total Current Assets' in balance_sheet.index else None
    current_liabilities = balance_sheet.loc['Total Current Liabilities'].iloc[0] if 'Total Current Liabilities' in balance_sheet.index else None
    current_ratio = current_assets / current_liabilities if current_assets and current_liabilities else None

    # Otras Métricas
    volume = hist['Volume'].iloc[-1] if 'Volume' in hist else None
    market_cap = info.get('marketCap', None)  # Capitalización de Mercado
    beta = info.get('beta', None)  # Beta

    return {
        "Ratios de Valoración": {
            "Price Earnings Ratio": trailing_pe,
            "Dividend Yield": dividend_yield * 100 if dividend_yield else None,
            "Price to Book Value": price_to_book_value,
            "PEG Ratio": peg_ratio,
            "Price to Cash Flow Ratio": price_to_cash_flow,
            "EV/EBITDA": ev_to_ebitda
        },
        "Ratios de Rentabilidad": {
            "Return on Equity": return_on_equity,
            "Profit Margin": profit_margin,
            "Operating Margin (ttm)": operating_margin,
            "Payout Ratio": payout_ratio
        },
        "Ratios de Liquidez y Solvencia": {
            "Current Ratio (mrq)": current_ratio
        },
        "Otras Métricas": {
            "Volumen": volume,
            "Earnings Per Share (EPS)": eps,
            "Capitalización de Mercado": market_cap,
            "Beta": beta
        }
    }
        # Obtener el ticker del usuario
        ticker = st.text_input("Introduce el ticker de la acción (por ejemplo, 'AAPL'):", value='AAPL')

        if ticker:
            try:
                # Calcular ratios financieros
                ratios = calculate_financial_ratios(ticker)
                
                for category, metrics in ratios.items():
                    st.write(f"### {category}")
                    for metric, value in metrics.items():
                        st.write(f"**{metric}:** {value if value is not None else 'N/A'}")
            
            except ValueError as e:
                st.error(str(e))

    elif submenu_acciones == "Riesgo":
        st.subheader("Riesgo")
        # Agrega aquí tu código para el análisis de riesgo
        st.write("Aquí puedes agregar el código para el análisis de riesgo.")

elif menu == "Gestión de Carteras":
    st.title("Gestión de Carteras")
    submenu_gestion = st.sidebar.selectbox("Selecciona una opción", ["Análisis de Carteras", "Optimización de Carteras"])

    if submenu_gestion == "Análisis de Carteras":
        st.subheader("Análisis de Carteras")
        # Agrega aquí tu código para el análisis de carteras
        st.write("Aquí puedes agregar el código para el análisis de carteras.")

    elif submenu_gestion == "Optimización de Carteras":
        st.subheader("Optimización de Carteras")
        # Agrega aquí tu código para la optimización de carteras
        st.write("Aquí puedes agregar el código para la optimización de carteras.")

    

        
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
