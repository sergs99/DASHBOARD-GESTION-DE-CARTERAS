import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
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
            @st.cache_data
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
            # Asegúrate de que info se obtiene correctamente
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

if submenu_acciones == "Riesgo":
        st.subheader("Análisis de Riesgo")

        # Entradas de usuario
        ticker = st.text_input("Símbolo bursátil:", value='AAPL')
        start_date = st.date_input('Fecha de inicio', (datetime.today() - timedelta(days=252)).date())
        end_date = st.date_input('Fecha de fin', datetime.today().date())
        market_ticker = st.text_input("Símbolo del Mercado:", value='^GSPC')

        if st.button('Calcular'):
            try:
                # Obtener datos históricos
                hist, _ = get_stock_data(ticker, start_date, end_date)
                hist['Returns'] = hist['Close'].pct_change().dropna()
                market_hist, _ = get_stock_data(market_ticker, start_date, end_date)
                market_hist['Returns'] = market_hist['Close'].pct_change().dropna()

                # Calcular métricas de riesgo
                var = calculate_var(hist['Returns'])
                cvar = calculate_cvar(hist['Returns'])
                volatility = calculate_volatility(hist['Returns'])
                drawdown = calculate_drawdown(hist['Returns'])
                beta = calculate_beta(hist['Returns'], market_hist['Returns'])
                sharpe_ratio = calculate_sharpe_ratio(hist['Returns'])
                sortino_ratio = calculate_sortino_ratio(hist['Returns'])
                variance = calculate_variance(hist['Returns'])
                kurtosis = calculate_kurtosis(hist['Returns'])
                skewness = calculate_skewness(hist['Returns'])

                # Mostrar métricas de riesgo
                st.write(f"Valor en Riesgo (VaR): {var:.2%}")
                st.write(f"Valor en Riesgo Condicional (CVaR): {cvar:.2%}")
                st.write(f"Volatilidad: {volatility:.2%}")
                st.write(f"Drawdown Máximo: {drawdown.min():.2%}")
                st.write(f"Beta: {beta:.2f}")
                st.write(f"Ratio Sharpe: {sharpe_ratio:.2f}")
                st.write(f"Ratio Sortino: {sortino_ratio:.2f}")
                st.write(f"Varianza: {variance:.2%}")
                st.write(f"Kurtosis: {kurtosis:.2f}")
                st.write(f"Sesgo: {skewness:.2f}")

                # Gráfico de Drawdown
                st.line_chart(drawdown, use_container_width=True)
                
                # Gráfico de Retornos
                plot_metrics(hist['Returns'], market_hist['Returns'])

            except Exception as e:
                st.error(f"Ocurrió un error: {e}")

# Si la opción seleccionada es "Gestión de Carteras"
elif menu == "Gestión de Carteras":
    st.subheader("Gestión de Carteras")
    # Aquí puedes agregar la funcionalidad para la gestión de carteras.
    pass
