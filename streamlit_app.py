import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import ta
import plotly.graph_objs as go
from datetime import datetime, timedelta

def app():
    # Título de la aplicación
    st.title("Aplicación de Análisis Financiero")

    # Menú desplegable en la barra lateral
    menu = st.sidebar.selectbox(
        "Selecciona una sección",
        ["Análisis de acciones", "Gestión de carteras"]
    )
def get_stock_data(ticker, start_date, end_date):
    stock = yf.Ticker(ticker)
    hist = stock.history(start=start_date, end=end_date)
    info = stock.info
    return hist, info
    if menu == "Análisis de acciones":
        handle_stock_analysis()
    elif menu == "Gestión de carteras":
        handle_portfolio_management()

def handle_stock_analysis():
    # Submenú para Análisis de acciones
    submenu = st.sidebar.selectbox(
        "Selecciona un tipo de análisis",
        ["Análisis Técnico", "Análisis Fundamental", "Análisis de Riesgo"]
    )

    if submenu == "Análisis Técnico":
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

    elif submenu == "Análisis Fundamental":
        st.subheader("Análisis Fundamental")
        # Aquí va el código para el análisis fundamental
        st.write("Aquí puedes implementar el análisis fundamental.")

    elif submenu == "Análisis de Riesgo":
        st.subheader("Análisis de Riesgo")
        # Aquí va el código para el análisis de riesgo
        st.write("Aquí puedes implementar el análisis de riesgo.")

def handle_portfolio_management():
    # Submenú para Gestión de carteras
    submenu = st.sidebar.selectbox(
        "Selecciona una opción de gestión de carteras",
        ["Análisis de Cartera", "Optimización de Cartera"]
    )

    if submenu == "Análisis de Cartera":
        st.subheader("Análisis de Cartera")
        # Aquí va el código para el análisis de cartera
        st.write("Aquí puedes implementar el análisis de cartera.")

    elif submenu == "Optimización de Cartera":
        st.subheader("Optimización de Cartera")
        # Aquí va el código para la optimización de cartera
        st.write("Aquí puedes implementar la optimización de cartera.")



if __name__ == "__main__":
    app()
