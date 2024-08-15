import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import ta
import plotly.graph_objs as go
from datetime import datetime, timedelta

# Configuración básica de la aplicación
st.set_page_config(page_title="Análisis Financiero", layout="wide")

# Título de la aplicación
st.title("Herramientas de Análisis Financiero")

# Menú principal
menu = st.sidebar.selectbox(
    "Seleccione una categoría",
    ("Acciones", "Gestión de Carteras")
)

# Función para obtener datos de la acción
def get_stock_data(ticker, start_date, end_date):
    stock = yf.Ticker(ticker)
    hist = stock.history(start=start_date, end=end_date)
    info = stock.info
    return hist, info

# Submenú en la categoría "Acciones"
if menu == "Acciones":
    submenu_acciones = st.sidebar.selectbox(
        "Seleccione un análisis para acciones",
        ("Análisis Técnico", "Análisis Fundamental", "Riesgo")
    )
    
    if submenu_acciones == "Análisis Técnico":
        st.subheader("Análisis Técnico")
        
        # Entradas de usuario
        ticker = st.text_input("Símbolo bursátil:", value='AAPL')
        start_date = st.date_input('Fecha de inicio', (datetime.today() - timedelta(days=252)).date())
        end_date = st.date_input('Fecha de fin', datetime.today().date())

        try:
            hist, info = get_stock_data(ticker, start_date, end_date)

            # Función para calcular indicadores técnicos
            @st.cache
            def calculate_technical_indicators(hist):
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
                data['Momentum'] = ta.momentum.roc(data['Close'], window=10)
                macd = ta.trend.MACD(data['Close'])
                data['MACD'] = macd.macd()
                data['MACD_Signal'] = macd.macd_signal()
                data['MACD_Histogram'] = macd.macd_diff()
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
            
            # Calcular indicadores técnicos
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
                title_font=dict(size=18, color='white'),
                xaxis_title='Fecha',
                xaxis_title_font=dict(size=14, color='white'),
                yaxis_title='Precio',
                yaxis_title_font=dict(size=14, color='white'),
                plot_bgcolor='black',
                paper_bgcolor='black',
                font=dict(color='white'),
                xaxis=dict(gridcolor='grey', zerolinecolor='grey'),
                yaxis=dict(gridcolor='grey', zerolinecolor='grey')
            )
            st.plotly_chart(price_fig)

            # Gráfico de Volumen
            volume_fig = go.Figure()
            volume_fig.add_trace(go.Bar(x=hist.index, y=hist['Volume'], name='Volumen de Negociación', marker_color='rgba(255, 87, 34, 0.8)'))
            volume_fig.update_layout(
                title=f'Volumen de Negociación de {ticker}',
                title_font=dict(size=18, color='white'),
                xaxis_title='Fecha',
                xaxis_title_font=dict(size=14, color='white'),
                yaxis_title='Volumen',
                yaxis_title_font=dict(size=14, color='white'),
                plot_bgcolor='black',
                paper_bgcolor='black',
                font=dict(color='white'),
                xaxis=dict(gridcolor='grey', zerolinecolor='grey'),
                yaxis=dict(gridcolor='grey', zerolinecolor='grey')
            )
            st.plotly_chart(volume_fig)

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
                title_font=dict(size=18, color='white'),
                xaxis_title='Fecha',
                xaxis_title_font=dict(size=14, color='white'),
                yaxis_title='Precio',
                yaxis_title_font=dict(size=14, color='white'),
                plot_bgcolor='black',
                paper_bgcolor='black',
                font=dict(color='white'),
                xaxis=dict(gridcolor='grey', zerolinecolor='grey'),
                yaxis=dict(gridcolor='grey', zerolinecolor='grey')
            )
            st.plotly_chart(bollinger_fig)

            # MACD
            macd_fig = go.Figure()
            macd_fig.add_trace(go.Scatter(x=data.index, y=data['MACD'], mode='lines', name='MACD', line=dict(color='blue')))
            macd_fig.add_trace(go.Scatter(x=data.index, y=data['MACD_Signal'], mode='lines', name='Línea de Señal', line=dict(color='red')))
            macd_fig.add_trace(go.Bar(x=data.index, y=data['MACD_Histogram'], name='Histograma', marker_color='rgba(255, 193, 7, 0.5)'))
            macd_fig.update_layout(
                title=f'MACD de {ticker}',
                title_font=dict(size=18, color='white'),
                xaxis_title='Fecha',
                xaxis_title_font=dict(size=14, color='white'),
                yaxis_title='MACD',
                yaxis_title_font=dict(size=14, color='white'),
                plot_bgcolor='black',
                paper_bgcolor='black',
                font=dict(color='white'),
                xaxis=dict(gridcolor='grey', zerolinecolor='grey'),
                yaxis=dict(gridcolor='grey', zerolinecolor='grey')
            )
            st.plotly_chart(macd_fig)

            # RSI
            rsi_fig = go.Figure()
            rsi_fig.add_trace(go.Scatter(x=data.index, y=data['RSI'], mode='lines', name='RSI', line=dict(color='blue')))
            rsi_fig.add_hline(y=70, line=dict(color='red', dash='dash'), name='Sobrecompra (70)')
            rsi_fig.add_hline(y=30, line=dict(color='green', dash='dash'), name='Sobreventa (30)')
            rsi_fig.update_layout(
                title=f'RSI de {ticker}',
                title_font=dict(size=18, color='white'),
                xaxis_title='Fecha',
                xaxis_title_font=dict(size=14, color='white'),
                yaxis_title='RSI',
                yaxis_title_font=dict(size=14, color='white'),
                plot_bgcolor='black',
                paper_bgcolor='black',
                font=dict(color='white'),
                xaxis=dict(gridcolor='grey', zerolinecolor='grey'),
                yaxis=dict(gridcolor='grey', zerolinecolor='grey')
            )
            st.plotly_chart(rsi_fig)

            # Stochastic Oscillator
            stoch_fig = go.Figure()
            stoch_fig.add_trace(go.Scatter(x=data.index, y=data['Stoch_K'], mode='lines', name='%K', line=dict(color='blue')))
            stoch_fig.add_trace(go.Scatter(x=data.index, y=data['Stoch_D'], mode='lines', name='%D', line=dict(color='red')))
            stoch_fig.add_hline(y=80, line=dict(color='red', dash='dash'), name='Sobrecompra (80)')
            stoch_fig.add_hline(y=20, line=dict(color='green', dash='dash'), name='Sobreventa (20)')
            stoch_fig.update_layout(
                title=f'Indicador Estocástico de {ticker}',
                title_font=dict(size=18, color='white'),
                xaxis_title='Fecha',
                xaxis_title_font=dict(size=14, color='white'),
                yaxis_title='Estocástico',
                yaxis_title_font=dict(size=14, color='white'),
                plot_bgcolor='black',
                paper_bgcolor='black',
                font=dict(color='white'),
                xaxis=dict(gridcolor='grey', zerolinecolor='grey'),
                yaxis=dict(gridcolor='grey', zerolinecolor='grey')
            )
            st.plotly_chart(stoch_fig)

            # ADX
            adx_fig = go.Figure()
            adx_fig.add_trace(go.Scatter(x=data.index, y=data['ADX'], mode='lines', name='ADX', line=dict(color='blue')))
            adx_fig.add_trace(go.Scatter(x=data.index, y=data['ADX_Pos'], mode='lines', name='+DI', line=dict(color='green')))
            adx_fig.add_trace(go.Scatter(x=data.index, y=data['ADX_Neg'], mode='lines', name='–DI', line=dict(color='red')))
            adx_fig.update_layout(
                title=f'ADX de {ticker}',
                title_font=dict(size=18, color='white'),
                xaxis_title='Fecha',
                xaxis_title_font=dict(size=14, color='white'),
                yaxis_title='ADX',
                yaxis_title_font=dict(size=14, color='white'),
                plot_bgcolor='black',
                paper_bgcolor='black',
                font=dict(color='white'),
                xaxis=dict(gridcolor='grey', zerolinecolor='grey'),
                yaxis=dict(gridcolor='grey', zerolinecolor='grey')
            )
            st.plotly_chart(adx_fig)

            # CCI
            cci_fig = go.Figure()
            cci_fig.add_trace(go.Scatter(x=data.index, y=data['CCI'], mode='lines', name='CCI', line=dict(color='blue')))
            cci_fig.add_hline(y=100, line=dict(color='red', dash='dash'), name='Sobrecompra (100)')
            cci_fig.add_hline(y=-100, line=dict(color='green', dash='dash'), name='Sobreventa (-100)')
            cci_fig.update_layout(
                title=f'CCI de {ticker}',
                title_font=dict(size=18, color='white'),
                xaxis_title='Fecha',
                xaxis_title_font=dict(size=14, color='white'),
                yaxis_title='CCI',
                yaxis_title_font=dict(size=14, color='white'),
                plot_bgcolor='black',
                paper_bgcolor='black',
                font=dict(color='white'),
                xaxis=dict(gridcolor='grey', zerolinecolor='grey'),
                yaxis=dict(gridcolor='grey', zerolinecolor='grey')
            )
            st.plotly_chart(cci_fig)

            # OBV
            obv_fig = go.Figure()
            obv_fig.add_trace(go.Scatter(x=data.index, y=data['OBV'], mode='lines', name='OBV', line=dict(color='blue')))
            obv_fig.update_layout(
                title=f'OBV de {ticker}',
                title_font=dict(size=18, color='white'),
                xaxis_title='Fecha',
                xaxis_title_font=dict(size=14, color='white'),
                yaxis_title='OBV',
                yaxis_title_font=dict(size=14, color='white'),
                plot_bgcolor='black',
                paper_bgcolor='black',
                font=dict(color='white'),
                xaxis=dict(gridcolor='grey', zerolinecolor='grey'),
                yaxis=dict(gridcolor='grey', zerolinecolor='grey')
            )
            st.plotly_chart(obv_fig)

            # VWAP
            vwap_fig = go.Figure()
            vwap_fig.add_trace(go.Scatter(x=data.index, y=data['VWAP'], mode='lines', name='VWAP', line=dict(color='blue')))
            vwap_fig.update_layout(
                title=f'VWAP de {ticker}',
                title_font=dict(size=18, color='white'),
                xaxis_title='Fecha',
                xaxis_title_font=dict(size=14, color='white'),
                yaxis_title='VWAP',
                yaxis_title_font=dict(size=14, color='white'),
                plot_bgcolor='black',
                paper_bgcolor='black',
                font=dict(color='white'),
                xaxis=dict(gridcolor='grey', zerolinecolor='grey'),
                yaxis=dict(gridcolor='grey', zerolinecolor='grey')
            )
            st.plotly_chart(vwap_fig)

        except Exception as e:
            st.error(f"Ocurrió un error: {e}")
if submenu_acciones == "Análisis Fundamental":
        st.subheader("Análisis Fundamental")
        
        # Entradas de usuario
        ticker = st.text_input("Símbolo bursátil:", value='AAPL')
        start_date = st.date_input('Fecha de inicio', (datetime.today() - timedelta(days=252)).date())
        end_date = st.date_input('Fecha de fin', datetime.today().date())

        try:
            # Obtener datos
            hist, info = get_stock_data(ticker, start_date, end_date)

            if info is None:
                st.error("No se pudo obtener información fundamental para el símbolo proporcionado.")
            else:
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

        except Exception as e:
            st.error(f"Ocurrió un error: {e}")

    elif submenu_acciones == "Riesgo":
        st.subheader("Análisis de Riesgo")
        # Aquí puedes agregar el análisis de riesgo y el resto de la implementación.
        pass


# Función para validar los pesos
def validate_weights(weights, num_assets):
    if abs(sum(weights) - 1) > 0.001:
        raise ValueError("Los pesos no suman 1.")
    if any(w < 0 for w in weights):
        raise ValueError("Los pesos no pueden ser negativos.")
    if len(weights) != num_assets:
        raise ValueError("El número de pesos no coincide con el número de activos.")
    return weights

# Función para descargar datos históricos
def download_data(tickers, benchmark_ticker):
    data = pd.DataFrame()
    for t in tickers + [benchmark_ticker]:
        try:
            data[t] = yf.download(t, start='2015-01-01')['Adj Close']
        except Exception as e:
            print(f"Error descargando datos para {t}: {e}")
            st.error(f"Error descargando datos para {t}: {e}")
            raise
    if data.empty:
        raise ValueError("No se pudo descargar datos para los tickers proporcionados.")
    return data

# Función para calcular rendimientos logarítmicos
def calculate_log_returns(data):
    return np.log(1 + data.pct_change()).dropna()

# Función para calcular estadísticas del portafolio
def portfolio_stats(weights, log_returns):
    port_ret = np.sum(log_returns.mean() * weights) * 252
    port_var = np.sqrt(np.dot(weights.T, np.dot(log_returns.cov() * 252, weights)))
    sharpe = port_ret / port_var
    return {'Return': port_ret, 'Volatility': port_var, 'Sharpe': sharpe}

# Función para calcular VaR y CVaR
def portfolio_risk_measures(weights, log_returns, alpha=0.05):
    port_returns = np.dot(log_returns, weights)
    var = np.percentile(port_returns, 100 * alpha)
    cvar = port_returns[port_returns <= var].mean()
    return {'VaR': var, 'CVaR': cvar}

# Función para calcular Ratio de Sortino
def sortino_ratio(returns, target_return=0):
    downside_returns = returns[returns < target_return]
    downside_volatility = np.sqrt((downside_returns**2).mean()) * np.sqrt(252)
    return (returns.mean() - target_return) / downside_volatility

# Función para calcular Desviación Media
def mean_absolute_deviation(returns):
    return np.mean(np.abs(returns - np.mean(returns))) * np.sqrt(252)

# Función para calcular Drawdown Máximo
def max_drawdown(returns):
    cumulative_returns = np.cumsum(returns)
    peaks = np.maximum.accumulate(cumulative_returns)
    drawdowns = (peaks - cumulative_returns) / peaks
    return np.max(drawdowns) * 100

# Función para calcular Ratio de Calmar
def calmar_ratio(returns):
    return np.mean(returns) / max_drawdown(returns)

# Función para calcular Beta
def calculate_beta(portfolio_returns, benchmark_returns):
    covariance = np.cov(portfolio_returns, benchmark_returns)[0, 1]
    variance = np.var(benchmark_returns)
    return covariance / variance

# Función para calcular Ratio de Treynor
def treynor_ratio(weights, log_returns, benchmark_returns, risk_free_rate=0.02):
    portfolio_returns = np.dot(log_returns, weights)
    beta = calculate_beta(portfolio_returns, benchmark_returns)
    portfolio_return = np.mean(portfolio_returns) * 252
    excess_return = portfolio_return - risk_free_rate
    return excess_return / beta

# Función para calcular Ratio de Información
def information_ratio(portfolio_returns, benchmark_returns):
    excess_returns = portfolio_returns - benchmark_returns
    tracking_error = np.std(excess_returns) * np.sqrt(252)
    average_excess_return = np.mean(excess_returns) * 252
    return average_excess_return / tracking_error

# Función para calcular Alpha de Jensen
def jensen_alpha(weights, log_returns, benchmark_returns, risk_free_rate=0.02):
    portfolio_returns = np.dot(log_returns, weights)
    beta = calculate_beta(portfolio_returns, benchmark_returns)
    portfolio_return = np.mean(portfolio_returns) * 252
    benchmark_return = np.mean(benchmark_returns) * 252
    alpha = portfolio_return - (risk_free_rate + beta * (benchmark_return - risk_free_rate))
    return alpha

# Función para calcular Ratio de Omega
def omega_ratio(portfolio_returns, risk_free_rate=0.02):
    threshold = np.mean(portfolio_returns) - risk_free_rate
    gain = portfolio_returns[portfolio_returns > threshold].sum()
    loss = -portfolio_returns[portfolio_returns <= threshold].sum()
    return gain / loss

# Función para análisis de sensibilidad (Stress Testing)
def stress_test(log_returns, weights):
    stressed_returns = log_returns * 1.10
    stressed_portfolio_returns = np.dot(stressed_returns, weights)
    return {
        'Stressed VaR': np.percentile(stressed_portfolio_returns, 5),
        'Stressed CVaR': stressed_portfolio_returns[stressed_portfolio_returns <= np.percentile(stressed_portfolio_returns, 5)].mean()
    }

# Aplicación Streamlit
def app():
    st.title("Aplicación de Análisis Financiero")
    
    # Menú desplegable
    menu = st.sidebar.selectbox(
        "Selecciona una sección",
        ["Análisis de acciones", "Gestión de carteras"]
    )
    
    if menu == "Gestión de carteras":
        st.subheader("Gestión de Carteras")
        portfolio_option = st.sidebar.selectbox(
            "Selecciona una opción de gestión de carteras",
            ["Análisis de cartera", "Optimización de cartera"]
        )
        
        if portfolio_option == "Análisis de cartera":
            st.subheader("Análisis de Cartera")
            st.write("Por favor, ingresa los datos necesarios para el análisis de cartera.")
            
            try:
                # Solicitar tickers, pesos y benchmark al usuario
                tickers_input = st.text_input("Ingresa los tickers de las acciones, separados por comas:")
                tickers = [ticker.strip().upper() for ticker in tickers_input.split(',') if ticker.strip()]
                
                weights_input = st.text_input("Ingresa los pesos correspondientes (deben sumar 1), separados por comas:")
                weights = [float(weight) for weight in weights_input.split(',') if weight.strip()]
                
                benchmark_ticker = st.text_input("Ingresa el ticker del benchmark (por ejemplo, ^GSPC):").strip().upper()
                
                # Verifica el número de tickers y pesos
                if len(weights) != len(tickers):
                    st.error(f"El número de pesos ({len(weights)}) debe coincidir con el número de tickers ({len(tickers)}).")
                else:
                    weights = validate_weights(weights, len(tickers))

                    # Descargar datos históricos
                    data = download_data(tickers, benchmark_ticker)

                    # Calcular rendimientos logarítmicos solo para los tickers de las acciones
                    log_returns = calculate_log_returns(data[tickers])

                    # Obtener los rendimientos del benchmark
                    benchmark_returns = np.log(1 + data[benchmark_ticker].pct_change()).dropna()

                    # Alinear las fechas del benchmark con los datos del portafolio
                    common_dates = log_returns.index.intersection(benchmark_returns.index)
                    log_returns = log_returns.loc[common_dates]
                    benchmark_returns = benchmark_returns.loc[common_dates]

                    # Convertir pesos a un array numpy
                    weights = np.array(weights)

                    # Calcular estadísticas del portafolio
                    portfolio_returns = np.dot(log_returns, weights)
                    optimal_stats = portfolio_stats(weights, log_returns)
                    optimal_risk_measures = portfolio_risk_measures(weights, log_returns)
                    sortino = sortino_ratio(portfolio_returns)
                    mad = mean_absolute_deviation(portfolio_returns)
                    max_dd = max_drawdown(portfolio_returns)
                    calmar = calmar_ratio(portfolio_returns)
                    treynor = treynor_ratio(weights, log_returns, benchmark_returns)
                    info_ratio = information_ratio(portfolio_returns, benchmark_returns)
                    alpha = jensen_alpha(weights, log_returns, benchmark_returns)
                    omega = omega_ratio(portfolio_returns)
                    stressed_risk = stress_test(log_returns, weights)

                    # Resultados
                    st.write("Retorno óptimo de la cartera: ", round(optimal_stats['Return'] * 100, 4))
                    st.write("Volatilidad óptima de la cartera: ", round(optimal_stats['Volatility'] * 100, 4))
                    st.write("Ratio Sharpe óptimo de la cartera: ", round(optimal_stats['Sharpe'], 4))
                    st.write("VaR (5%): ", round(optimal_risk_measures['VaR'] * 100, 4))
                    st.write("CVaR (5%): ", round(optimal_risk_measures['CVaR'] * 100, 4))
                    st.write("Ratio de Sortino: ", round(sortino, 4))
                    st.write("Desviación Media: ", round(mad * 100, 4))
                    st.write("Drawdown Máximo: ", round(max_dd, 4))
                    st.write("Ratio de Calmar: ", round(calmar, 4))
                    st.write("Ratio de Treynor: ", round(treynor, 4))
                    st.write("Ratio de Información: ", round(info_ratio, 4))
                    st.write("Alpha de Jensen: ", round(alpha, 4))
                    st.write("Ratio de Omega: ", round(omega, 4))
                    st.write("VaR (5%) en escenario estresado: ", round(stressed_risk['Stressed VaR'] * 100, 4))
                    st.write("CVaR (5%) en escenario estresado: ", round(stressed_risk['Stressed CVaR'] * 100, 4))

                    # Graficar la evolución de los precios
                    st.subheader("Evolución de Precios Normalizados")
                    fig, ax = plt.subplots(figsize=(12, 6))
                    (data / data.iloc[0] * 100).plot(ax=ax)
                    ax.set_xlabel('Fecha')
                    ax.set_ylabel('Precio Normalizado')
                    ax.set_title('Evolución de Precios Normalizados')
                    st.pyplot(fig)

                    # Histograma de retornos
                    st.subheader("Distribución de Retornos Diarios")
                    fig, ax = plt.subplots(figsize=(12, 6))
                    ax.hist(portfolio_returns, bins=50, alpha=0.75, label='Retornos logarítmicos')
                    ax.set_xlabel('Retornos logarítmicos diarios')
                    ax.set_ylabel('Frecuencia')
                    ax.set_title('Distribución de Retornos Diarios')
                    ax.legend()
                    st.pyplot(fig)

                    # Heatmap de la matriz de correlación
                    st.subheader("Matriz de Correlación")
                    fig, ax = plt.subplots(figsize=(12, 6))
                    sns.heatmap(log_returns.corr(), annot=True, linewidths=1, cmap='coolwarm', ax=ax)
                    ax.set_title("Matriz de Correlación")
                    st.pyplot(fig)

            except Exception as e:
                st.error(f"Ocurrió un error: {e}")

        elif portfolio_option == "Optimización de cartera":
            st.subheader("Optimización de Cartera")
            st.write("Aquí va el contenido para la optimización de cartera.")
            # Agrega tu código y widgets aquí

if __name__ == "__main__":
    app()


