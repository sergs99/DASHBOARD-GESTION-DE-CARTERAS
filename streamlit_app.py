import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import ta
import plotly.graph_objs as go
from datetime import datetime, timedelta

# Configuración básica de la aplicación
st.set_page_config(page_title="Análisis Financiero", layout="wide")

# Título de la aplicación
st.title("Herramientas de Análisis Financiero")

# Menú principal
menu = st.sidebar.selectbox("Seleccione una categoría", ("Acciones", "Gestión de Carteras"))

def get_stock_data(ticker, start_date, end_date):
    """Obtiene los datos históricos y la información fundamental de una acción."""
    stock = yf.Ticker(ticker)
    hist = stock.history(start=start_date, end=end_date)
    info = stock.info
    return hist, info

def calculate_technical_indicators(hist):
    """Calcula indicadores técnicos para los datos históricos de una acción."""
    data = hist.copy()
    data['SMA_50'] = ta.trend.sma_indicator(data['Close'], window=50)
    data['SMA_100'] = ta.trend.sma_indicator(data['Close'], window=100)
    data['SMA_200'] = ta.trend.sma_indicator(data['Close'], window=200)
    data['EMA_20'] = ta.trend.ema_indicator(data['Close'], window=20)
    data['EMA_50'] = ta.trend.ema_indicator(data['Close'], window=50)
    bollinger = ta.volatility.BollingerBands(close=data['Close'])
    data['BB_High'] = bollinger.bollinger_hband()
    data['BB_Low'] = bollinger.bollinger_lband()
    data['BB_Middle'] = bollinger.bollinger_mavg()
    stoch = ta.momentum.StochasticOscillator(high=data['High'], low=data['Low'], close=data['Close'])
    data['Stoch_K'] = stoch.stoch()
    data['Stoch_D'] = stoch.stoch_signal()
    data['RSI'] = ta.momentum.RSIIndicator(data['Close']).rsi()
    data['MACD'] = ta.trend.MACD(data['Close']).macd()
    data['MACD_Signal'] = ta.trend.MACD(data['Close']).macd_signal()
    data['MACD_Histogram'] = ta.trend.MACD(data['Close']).macd_diff()
    adx = ta.trend.ADXIndicator(high=data['High'], low=data['Low'], close=data['Close'])
    data['ADX'] = adx.adx()
    data['ADX_Pos'] = adx.adx_pos()
    data['ADX_Neg'] = adx.adx_neg()
    data['CCI'] = ta.trend.CCIIndicator(high=data['High'], low=data['Low'], close=data['Close']).cci()
    obv = ta.volume.OnBalanceVolumeIndicator(close=data['Close'], volume=data['Volume'])
    data['OBV'] = obv.on_balance_volume()
    vwap = ta.volume.VolumeWeightedAveragePrice(high=data['High'], low=data['Low'], close=data['Close'], volume=data['Volume'])
    data['VWAP'] = vwap.volume_weighted_average_price()
    return data

if menu == "Acciones":
    submenu_acciones = st.sidebar.selectbox("Seleccione un análisis para acciones", ("Análisis Técnico", "Análisis Fundamental", "Riesgo"))
    
    if submenu_acciones == "Análisis Técnico":
        st.subheader("Análisis Técnico")
        ticker = st.text_input("Símbolo bursátil:", value='AAPL')
        start_date = st.date_input('Fecha de inicio', (datetime.today() - timedelta(days=252)).date())
        end_date = st.date_input('Fecha de fin', datetime.today().date())

        try:
            hist, info = get_stock_data(ticker, start_date, end_date)
            data = calculate_technical_indicators(hist)

            # Gráfico de Velas
            price_fig = go.Figure(data=[go.Candlestick(
                x=hist.index,
                open=hist['Open'],
                high=hist['High'],
                low=hist['Low'],
                close=hist['Close'],
                increasing_line_color='lime',
                decreasing_line_color='red',
                name='Candlestick'
            )])
            price_fig.update_layout(
                title=f'Gráfico de Velas de {ticker}',
                xaxis_title='Fecha',
                yaxis_title='Precio',
                plot_bgcolor='black',
                paper_bgcolor='black',
                font=dict(color='white')
            )
            st.plotly_chart(price_fig)

            # Bandas de Bollinger
            bollinger_fig = go.Figure()
            bollinger_fig.add_trace(go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                increasing_line_color='lime',
                decreasing_line_color='red',
                name='Candlestick'
            ))
            bollinger_fig.add_trace(go.Scatter(x=data.index, y=data['BB_Middle'], mode='lines', name='Media Móvil', line=dict(color='cyan')))
            bollinger_fig.add_trace(go.Scatter(x=data.index, y=data['BB_High'], mode='lines', name='Banda Superior', line=dict(color='red')))
            bollinger_fig.add_trace(go.Scatter(x=data.index, y=data['BB_Low'], mode='lines', name='Banda Inferior', line=dict(color='green')))
            bollinger_fig.update_layout(
                title=f'Bandas de Bollinger de {ticker}',
                xaxis_title='Fecha',
                yaxis_title='Precio',
                plot_bgcolor='black',
                paper_bgcolor='black',
                font=dict(color='white')
            )
            st.plotly_chart(bollinger_fig)

        except Exception as e:
            st.error(f"Ocurrió un error: {e}")

    if submenu_acciones == "Análisis Fundamental":
        st.subheader("Análisis Fundamental")
        ticker = st.text_input("Símbolo bursátil:", value='AAPL')
        start_date = st.date_input('Fecha de inicio', (datetime.today() - timedelta(days=252)).date())
        end_date = st.date_input('Fecha de fin', datetime.today().date())

        try:
            hist, info = get_stock_data(ticker, start_date, end_date)

            if not info:
                st.error("No se pudo obtener información fundamental para el símbolo proporcionado.")
            else:
                fundamental_data = {
                    'Nombre': info.get('shortName', 'N/A'),
                    'Sector': info.get('sector', 'N/A'),
                    'Industria': info.get('industry', 'N/A'),
                    'Precio Actual': f"${info.get('currentPrice', 'N/A'):.2f}" if 'currentPrice' in info else 'N/A',
                    'Ratios de Valoración': {
                        'Price Earnings Ratio': info.get('trailingPE', 'N/A'),
                        'Dividend Yield': f"{info.get('dividendYield', 'N/A')*100:.2f}%" if 'dividendYield' in info else 'N/A',
                        'Price to Book Value': info.get('priceToBook', 'N/A'),
                        'PEG Ratio (5yr expected)': info.get('pegRatio', 'N/A'),
                        'Price to Cash Flow Ratio': info.get('priceToCashflow', 'N/A'),
                        'EV/EBITDA': info.get('enterpriseToEbitda', 'N/A')
                    },
                    'Ratios de Rentabilidad': {
                        'Return on Equity': f"{info.get('returnOnEquity', 'N/A')*100:.2f}%" if 'returnOnEquity' in info else 'N/A',
                        'Return on Assets': f"{info.get('returnOnAssets', 'N/A')*100:.2f}%" if 'returnOnAssets' in info else 'N/A',
                        'Return on Investment': f"{info.get('returnOnInvestment', 'N/A')*100:.2f}%" if 'returnOnInvestment' in info else 'N/A'
                    }
                }

                st.write("**Datos Fundamentales**")
                st.write(f"**Nombre**: {fundamental_data['Nombre']}")
                st.write(f"**Sector**: {fundamental_data['Sector']}")
                st.write(f"**Industria**: {fundamental_data['Industria']}")
                st.write(f"**Precio Actual**: {fundamental_data['Precio Actual']}")
                
                st.write("**Ratios de Valoración**")
                for ratio, value in fundamental_data['Ratios de Valoración'].items():
                    st.write(f"**{ratio}**: {value}")

                st.write("**Ratios de Rentabilidad**")
                for ratio, value in fundamental_data['Ratios de Rentabilidad'].items():
                    st.write(f"**{ratio}**: {value}")

        except Exception as e:
            st.error(f"Ocurrió un error: {e}")

import matplotlib.pyplot as plt
import seaborn as sns

# Función para la gestión de carteras
def portfolio_analysis(tickers, weights):
    """Realiza análisis de una cartera de inversión basada en los tickers y pesos proporcionados."""
    try:
        # Descargamos los datos históricos para todos los tickers
        data = {ticker: yf.Ticker(ticker).history(period='1y')['Close'] for ticker in tickers}
        prices = pd.DataFrame(data)

        # Calculamos los rendimientos diarios
        returns = prices.pct_change().dropna()
        
        # Rendimiento esperado de la cartera
        portfolio_return = np.dot(weights, returns.mean()) * 252  # Anualizado
        
        # Riesgo (desviación estándar) de la cartera
        portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))  # Anualizado

        # Optimización del portafolio
        num_portfolios = 10000
        results = np.zeros((num_portfolios, 3))
        for i in range(num_portfolios):
            w = np.random.random(len(tickers))
            w /= np.sum(w)
            returns_port = np.dot(w, returns.mean()) * 252
            risk_port = np.sqrt(np.dot(w.T, np.dot(returns.cov() * 252, w)))
            results[i, 0] = returns_port
            results[i, 1] = risk_port
            results[i, 2] = results[i, 0] / results[i, 1]  # Sharpe ratio

        results_df = pd.DataFrame(results, columns=['Return', 'Risk', 'Sharpe'])
        best_portfolio = results_df.iloc[results_df['Sharpe'].idxmax()]

        st.subheader("Análisis de Carteras")
        st.write(f"**Rendimiento Anualizado de la Cartera**: {portfolio_return:.2f}%")
        st.write(f"**Riesgo Anualizado de la Cartera**: {portfolio_risk:.2f}%")
        
        st.write("**Optimización de la Cartera**")
        st.write(f"**Mejor Sharpe Ratio**: {best_portfolio['Sharpe']:.2f}")
        st.write(f"**Rendimiento del Mejor Portafolio**: {best_portfolio['Return']:.2f}%")
        st.write(f"**Riesgo del Mejor Portafolio**: {best_portfolio['Risk']:.2f}%")
        
        # Gráfico del Frontera Eficiente
        plt.figure(figsize=(10, 6))
        plt.scatter(results[:, 1], results[:, 0], c=results[:, 2], cmap='viridis', marker='o')
        plt.colorbar(label='Sharpe Ratio')
        plt.title('Frontera Eficiente')
        plt.xlabel('Riesgo')
        plt.ylabel('Rendimiento')
        st.pyplot(plt)

    except Exception as e:
        st.error(f"Ocurrió un error en el análisis de carteras: {e}")

# Función para análisis de riesgo
def risk_analysis(ticker, start_date, end_date):
    """Realiza un análisis de riesgo basado en el VaR y CVaR de una acción."""
    try:
        hist, _ = get_stock_data(ticker, start_date, end_date)
        hist['Returns'] = hist['Close'].pct_change().dropna()
        
        # Valor en Riesgo (VaR)
        VaR = hist['Returns'].quantile(0.05)
        
        # Pérdida Esperada Condicional (CVaR)
        CVaR = hist['Returns'][hist['Returns'] <= VaR].mean()
        
        st.subheader("Análisis de Riesgo")
        st.write(f"**VaR (5% percentil)**: {VaR:.2%}")
        st.write(f"**CVaR (Pérdida Esperada Condicional)**: {CVaR:.2%}")

        # Gráfico de Rendimientos
        plt.figure(figsize=(10, 6))
        hist['Returns'].hist(bins=50, edgecolor='k', alpha=0.7)
        plt.axvline(VaR, color='r', linestyle='dashed', linewidth=1, label='VaR')
        plt.title(f'Distribución de Rendimientos de {ticker}')
        plt.xlabel('Rendimiento Diario')
        plt.ylabel('Frecuencia')
        plt.legend()
        st.pyplot(plt)
        
    except Exception as e:
        st.error(f"Ocurrió un error en el análisis de riesgo: {e}")

# Gestión de Carteras
if menu == "Gestión de Carteras":
    st.subheader("Gestión de Carteras de Inversión")
    tickers_input = st.text_input("Ingrese los símbolos de las acciones (separados por comas):", "AAPL,MSFT,GOOGL")
    weights_input = st.text_input("Ingrese los pesos de las acciones (separados por comas, deben sumar 1):", "0.4,0.4,0.2")
    
    tickers = [ticker.strip().upper() for ticker in tickers_input.split(',')]
    weights = np.array([float(weight.strip()) for weight in weights_input.split(',')])
    
    if len(tickers) != len(weights):
        st.error("El número de símbolos y pesos debe coincidir.")
    else:
        portfolio_analysis(tickers, weights)

# Análisis de Riesgo
if menu == "Riesgo":
    st.subheader("Análisis de Riesgo")
    ticker_risk = st.text_input("Símbolo bursátil para el análisis de riesgo:", value='AAPL')
    start_date_risk = st.date_input('Fecha de inicio', (datetime.today() - timedelta(days=252)).date())
    end_date_risk = st.date_input('Fecha de fin', datetime.today().date())
    
    if ticker_risk:
        risk_analysis(ticker_risk, start_date_risk, end_date_risk)

