import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

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
            st.error(f"Error descargando datos para {t}: {e}")
            return pd.DataFrame()
    if data.empty:
        st.error("No se pudo descargar datos para los tickers proporcionados.")
        return pd.DataFrame()
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

# Interfaz de usuario en Streamlit
st.title("Análisis y Optimización de Carteras")

# Solicitar tickers, pesos y benchmark al usuario
tickers_input = st.text_input("Ingresa los tickers de las acciones, separados por comas:", "AAPL, MSFT, GOOG")
tickers = [ticker.strip().upper() for ticker in tickers_input.split(',')]

weights_input = st.text_input("Ingresa los pesos correspondientes (deben sumar 1), separados por comas:", "0.4, 0.4, 0.2")
weights = [float(weight) for weight in weights_input.split(',')]

benchmark_ticker = st.text_input("Ingresa el ticker del benchmark (por ejemplo, ^GSPC):", "^GSPC").strip().upper()

try:
    # Verifica el número de tickers y pesos
    if len(weights) != len(tickers):
        st.error(f"El número de pesos ({len(weights)}) debe coincidir con el número de tickers ({len(tickers)}).")
    
    weights = validate_weights(weights, len(tickers))

    # Descargar datos históricos
    data = download_data(tickers, benchmark_ticker)

    if data.empty:
        st.stop()

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
    st.subheader("Estadísticas de la Cartera")
    st.write(f"Retorno óptimo de la cartera: {round(optimal_stats['Return'] * 100, 4)}%")
    st.write(f"Volatilidad óptima de la cartera: {round(optimal_stats['Volatility'] * 100, 4)}%")
    st.write(f"Ratio Sharpe óptimo de la cartera: {round(optimal_stats['Sharpe'], 4)}")
    st.write(f"VaR (5%): {round(optimal_risk_measures['VaR'] * 100, 4)}%")
    st.write(f"CVaR (5%): {round(optimal_risk_measures['CVaR'] * 100, 4)}%")
    st.write(f"Ratio de Sortino: {round(sortino, 4)}")
    st.write(f"Desviación Media: {round(mad * 100, 4)}%")
    st.write(f"Drawdown Máximo: {round(max_dd, 4)}%")
    st.write(f"Ratio de Calmar: {round(calmar, 4)}")
    st.write(f"Ratio de Treynor: {round(treynor, 4)}")
    st.write(f"Ratio de Información: {round(info_ratio, 4)}")
    st.write(f"Alpha de Jensen: {round(alpha, 4)}")
    st.write(f"Ratio de Omega: {round(omega, 4)}")
    st.write(f"VaR (5%) en escenario estresado: {round(stressed_risk['Stressed VaR'] * 100, 4)}%")
    st.write(f"CVaR (5%) en escenario estresado: {round(stressed_risk['Stressed CVaR'] * 100, 4)}%")

    # Graficar la evolución de los precios
    st.subheader("Evolución de Precios Normalizados")
    plt.figure(figsize=(12, 6))
    (data / data.iloc[0] * 100).plot()
    plt.xlabel('Fecha')
    plt.ylabel('Precio Normalizado')
    plt.title('Evolución de Precios Normalizados')
    st.pyplot(plt)

    # Histograma de retornos
    st.subheader("Distribución de Retornos Diarios")
    plt.figure(figsize=(12, 6))
    plt.hist(portfolio_returns, bins=50, alpha=0.75, label='Retornos logarítmicos')
    plt.xlabel('Retornos logarítmicos diarios')
    plt.ylabel('Frecuencia')
    plt.title('Distribución de Retornos Diarios')
    plt.legend()
    st.pyplot(plt)

    # Heatmap de la matriz de correlación
    st.subheader("Matriz de Correlación")
    plt.figure(figsize=(12, 6))
    sns.heatmap(log_returns.corr(), annot=True, linewidths=1, cmap='coolwarm')
    plt.title("Matriz de Correlación")
    st.pyplot(plt)

except ValueError as e:
    st.error(str(e))
