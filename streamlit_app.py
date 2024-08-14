import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import ta
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
            stoch_fig.add_hline(y=80, line=dict(color='grey', dash='dash'), name='Sobrecompra (80)')
            stoch_fig.add_hline(y=20, line=dict(color='grey', dash='dash'), name='Sobreventa (20)')
            stoch_fig.update_layout(
                title=f'Oscilador Estocástico de {ticker}',
                title_font=dict(size=18, color='white'),
                xaxis_title='Fecha',
                xaxis_title_font=dict(size=14, color='white'),
                yaxis_title='Oscilador',
                yaxis_title_font=dict(size=14, color='white'),
                plot_bgcolor='black',
                paper_bgcolor='black',
                font=dict(color='white'),
                xaxis=dict(gridcolor='grey', zerolinecolor='grey'),
                yaxis=dict(gridcolor='grey', zerolinecolor='grey')
            )
            st.plotly_chart(stoch_fig)

    elif submenu_acciones == "Análisis Fundamental":
        st.subheader("Análisis Fundamental")
        # Aquí puedes agregar el análisis fundamental y el resto de la implementación.
        pass

    elif submenu_acciones == "Riesgo":
        st.subheader("Análisis de Riesgo")
        # Aquí puedes agregar el análisis de riesgo y el resto de la implementación.
        pass

# Si la opción seleccionada es "Gestión de Carteras"
elif menu == "Gestión de Carteras":
    st.subheader("Gestión de Carteras")
    # Aquí puedes agregar la funcionalidad para la gestión de carteras.
    pass
