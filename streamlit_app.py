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
    
    # Obtener el ticker del usuario
    ticker = st.text_input("Introduce el ticker de la acción (por ejemplo, 'AAPL'):", value='AAPL')

    if ticker:
        # Descargar datos históricos y financieros
        stock = yf.Ticker(ticker)
        
        # Obtener el historial de precios
        hist = stock.history(period="1y")
        
        # Obtener información financiera
        info = stock.info
        
        # Mostrar algunos datos básicos
        st.write("### Datos Básicos:")
        st.write(f"**Nombre:** {info.get('longName', 'N/A')}")
        st.write(f"**Sector:** {info.get('sector', 'N/A')}")
        st.write(f"**Industria:** {info.get('industry', 'N/A')}")
        st.write(f"**Precio Actual:** {info.get('currentPrice', 'N/A'):.2f}")

        # Calcular y mostrar Dividend Yield
        dividend_yield = info.get('dividendYield', 'N/A')
        if dividend_yield != 'N/A':
            dividend_yield *= 100
        st.write(f"**Dividend Yield:** {dividend_yield:.2f}% (Óptimo: 2%-5%)" if dividend_yield != 'N/A' else "Dividend Yield: N/A")

        # Calcular y mostrar el Price to Earnings Ratio (PER)
        eps = info.get('forwardEps', 'N/A')
        current_price = info.get('currentPrice', 1)
        pe_ratio = eps / current_price if eps != 'N/A' and current_price != 0 else 'N/A'
        st.write(f"**Price to Earnings Ratio (PER):** {pe_ratio:.2f} (Óptimo: 15-25)" if pe_ratio != 'N/A' else "Price to Earnings Ratio (PER): N/A")

        # Calcular y mostrar el Price to Book Value (P/B)
        pb_ratio = info.get('priceToBook', 'N/A')
        st.write(f"**Price to Book Value (P/B):** {pb_ratio:.2f} (Óptimo: 1-3)" if pb_ratio != 'N/A' else "Price to Book Value (P/B): N/A")

        # Calcular y mostrar el PEG Ratio
        peg_ratio = info.get('pegRatio', 'N/A')
        st.write(f"**PEG Ratio:** {peg_ratio:.2f} (Óptimo: 0.5-1.5)" if peg_ratio != 'N/A' else "PEG Ratio: N/A")

        # Calcular y mostrar el Price to Cash Flow Ratio
        try:
            cash_flow = stock.cashflow.loc['Total Cash From Operating Activities'].iloc[0]
            shares_outstanding = info.get('sharesOutstanding', 1)
            price_to_cash_flow = current_price / (cash_flow / shares_outstanding)
            st.write(f"**Price to Cash Flow Ratio:** {price_to_cash_flow:.2f} (Óptimo: 5-15)")
        except (KeyError, IndexError):
            st.write("Datos de flujo de caja no disponibles.")

        # Calcular y mostrar el EV/EBITDA
        ev_to_ebitda = info.get('enterpriseToEbitda', 'N/A')
        st.write(f"**EV/EBITDA:** {ev_to_ebitda:.2f} (Óptimo: 8-12)" if ev_to_ebitda != 'N/A' else "EV/EBITDA: N/A")

        # Calcular y mostrar el Return on Equity (ROE)
        roe = info.get('returnOnEquity', 'N/A')
        st.write(f"**Return on Equity (ROE):** {roe:.2%} (Óptimo: 15%-20%)" if roe != 'N/A' else "Return on Equity (ROE): N/A")

        # Calcular y mostrar el Profit Margin
        profit_margin = info.get('profitMargins', 'N/A')
        st.write(f"**Profit Margin:** {profit_margin:.2%} (Óptimo: 10%-20%)" if profit_margin != 'N/A' else "Profit Margin: N/A")

        # Calcular y mostrar el Operating Margin
        operating_margin = info.get('operatingMargins', 'N/A')
        st.write(f"**Operating Margin (ttm):** {operating_margin:.2%} (Óptimo: 10%-20%)" if operating_margin != 'N/A' else "Operating Margin (ttm): N/A")

        # Calcular y mostrar el Payout Ratio
        payout_ratio = info.get('payoutRatio', 'N/A')
        st.write(f"**Payout Ratio:** {payout_ratio:.2%} (Óptimo: 30%-50%)" if payout_ratio != 'N/A' else "Payout Ratio: N/A")

        # Calcular y mostrar el Current Ratio
        try:
            balance_sheet = stock.balance_sheet
            current_assets = balance_sheet.loc['Total Current Assets'].iloc[0]
            current_liabilities = balance_sheet.loc['Total Current Liabilities'].iloc[0]
            current_ratio = current_assets / current_liabilities
            st.write(f"**Current Ratio (mrq):** {current_ratio:.2f} (Óptimo: 1.5-2.5)")
        except (KeyError, IndexError):
            st.write("Datos del balance general no disponibles.")
        
        # Calcular y mostrar el Quick Ratio
        try:
            quick_assets = balance_sheet.loc['Total Current Assets'].iloc[0] - balance_sheet.loc['Total Inventory'].iloc[0]
            quick_ratio = quick_assets / current_liabilities
            st.write(f"**Quick Ratio (Acid-Test Ratio):** {quick_ratio:.2f} (Óptimo: 1-2)")
        except (KeyError, IndexError):
            st.write("Datos para Quick Ratio no disponibles.")
        
        # Calcular y mostrar el Debt to Equity Ratio
        try:
            total_debt = balance_sheet.loc['Total Liabilities Net Minority Interest'].iloc[0]
            total_equity = balance_sheet.loc['Total Stockholder Equity'].iloc[0]
            debt_to_equity = total_debt / total_equity
            st.write(f"**Debt to Equity Ratio:** {debt_to_equity:.2f} (Óptimo: 0.5-1.0)")
        except (KeyError, IndexError):
            st.write("Datos para Debt to Equity Ratio no disponibles.")
        
        # Calcular y mostrar Free Cash Flow
        try:
            cash_flow_statement = stock.cashflow
            free_cash_flow = cash_flow_statement.loc['Free Cash Flow'].iloc[0]
            st.write(f"**Free Cash Flow:** {free_cash_flow:.2f} (Óptimo: Positivo y creciente)")
        except (KeyError, IndexError):
            st.write("Datos de flujo de caja libre no disponibles.")
        
        # Calcular y mostrar el Debt to EBITDA
        try:
            total_debt = balance_sheet.loc['Total Liabilities Net Minority Interest'].iloc[0]
            ebitda = stock.financials.loc['EBITDA'].iloc[0]
            debt_to_ebitda = total_debt / ebitda
            st.write(f"**Debt to EBITDA:** {debt_to_ebitda:.2f} (Óptimo: 1-3)")
        except (KeyError, IndexError):
            st.write("Datos para Debt to EBITDA no disponibles.")

        # Calcular y mostrar el Gross Margin
        gross_margin = info.get('grossMargins', 'N/A')
        st.write(f"**Gross Margin:** {gross_margin:.2%} (Óptimo: 20%-50%)" if gross_margin != 'N/A' else "Gross Margin: N/A")

        # Calcular y mostrar el Revenue Growth Rate
        try:
            revenue = stock.financials.loc['Total Revenue'].iloc[0]
            previous_revenue = stock.financials.loc['Total Revenue'].iloc[1]  # Asumir dos períodos para cálculo de crecimiento
            revenue_growth_rate = (revenue - previous_revenue) / previous_revenue * 100
            st.write(f"**Revenue Growth Rate:** {revenue_growth_rate:.2f}% (Óptimo: 5%-15%)")
        except (KeyError, IndexError):
            st.write("Datos para Revenue Growth Rate no disponibles.")
        
        # Calcular y mostrar el Earnings Growth Rate
        try:
            net_income = stock.financials.loc['Net Income'].iloc[0]
            previous_net_income = stock.financials.loc['Net Income'].iloc[1]  # Asumir dos períodos para cálculo de crecimiento
            earnings_growth_rate = (net_income - previous_net_income) / previous_net_income * 100
            st.write(f"**Earnings Growth Rate:** {earnings_growth_rate:.2f}% (Óptimo: 5%-15%)")
        except (KeyError, IndexError):
            st.write("Datos para Earnings Growth Rate no disponibles.")
        
        # Calcular y mostrar Market Capitalization
        market_cap = info.get('marketCap', 'N/A')
        st.write(f"**Market Capitalization:** {market_cap:.2e} (Óptimo: Depende del tipo de acción)" if market_cap != 'N/A' else "Market Capitalization: N/A")

        # Calcular y mostrar 52-Week High/Low
        try:
            high_52_week = info.get('fiftyTwoWeekHigh', 'N/A')
            low_52_week = info.get('fiftyTwoWeekLow', 'N/A')
            st.write(f"**52-Week High:** {high_52_week:.2f} (Óptimo: Evaluar en contexto)" if high_52_week != 'N/A' else "52-Week High: N/A")
            st.write(f"**52-Week Low:** {low_52_week:.2f} (Óptimo: Evaluar en contexto)" if low_52_week != 'N/A' else "52-Week Low: N/A")
        except KeyError:
            st.write("Datos para 52-Week High/Low no disponibles.")

        
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
