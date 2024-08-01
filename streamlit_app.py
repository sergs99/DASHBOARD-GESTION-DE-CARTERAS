import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from scipy.optimize import minimize
from datetime import datetime

def get_user_input():
    st.sidebar.header("Configuración de la Cartera")
    
    tickers_input = st.sidebar.text_input("Introduce los tickers de las acciones (separados por comas):").strip()
    weights_input = st.sidebar.text_input("Introduce los pesos de las acciones (separados por comas, deben sumar 1):").strip()
    
    if not tickers_input or not weights_input:
        st.error("Por favor, introduce tanto los tickers como los pesos.")
        return None, None, None
    
    tickers = [ticker.strip().upper() for ticker in tickers_input.split(',')]
    
    try:
        weights = np.array([float(weight.strip()) for weight in weights_input.split(',')])
    except ValueError:
        st.error("Los pesos deben ser números válidos.")
        return None, None, None
    
    if len(tickers) != len(weights):
        st.error("El número de tickers y pesos no coincide.")
        return None, None, None
    
    if not np.isclose(sum(weights), 1.0, atol=1e-5):
        st.error("La suma de los pesos debe ser aproximadamente igual a 1.0.")
        return None, None, None
    
    try:
        risk_free_rate = float(st.sidebar.text_input("Introduce la tasa libre de riesgo actual (como fracción, ej. 0.0234 para 2.34%):").strip())
    except ValueError:
        st.error("La tasa libre de riesgo debe ser un número válido.")
        return None, None, None

    return tickers, weights, risk_free_rate

def download_data(tickers_with_market):
    try:
        data = yf.download(tickers_with_market, start='2020-01-01', end=datetime.today().strftime('%Y-%m-%d'))['Adj Close']
    except Exception as e:
        st.error(f"Error al descargar datos: {e}")
        return None
    return data

def filter_valid_tickers(data):
    if data.empty:
        st.error("No se descargaron datos para los tickers proporcionados.")
        return data, []

    valid_tickers = [ticker for ticker in data.columns if not data[ticker].isnull().all()]
    if '^GSPC' not in valid_tickers:
        st.error("No se encontraron datos para el índice de mercado (^GSPC).")
        return data, []

    data = data[valid_tickers]
    return data, valid_tickers

def calculate_portfolio_metrics(tickers, weights):
    tickers_with_market = tickers + ['^GSPC']
    data = download_data(tickers_with_market)
    
    if data is None:
        return None, None, None, None, None, None, None, None
    
    data, valid_tickers = filter_valid_tickers(data)
    
    if data.empty:
        return None, None, None, None, None, None, None, None

    returns = data.pct_change(fill_method=None).dropna()
    
    if returns.shape[0] < 2:
        st.error("Los datos descargados no tienen suficientes retornos.")
        return None, None, None, None, None, None, None, None

    market_returns = returns['^GSPC']
    portfolio_returns = returns[tickers].dot(weights)
    
    if portfolio_returns.empty or len(portfolio_returns) < 2:
        st.error("Los datos de retornos de la cartera no tienen suficientes valores.")
        return None, None, None, None, None, None, None, None

    annualized_return = portfolio_returns.mean() * 252
    annualized_volatility = portfolio_returns.std() * np.sqrt(252)
    
    cumulative_return = (1 + portfolio_returns).prod() - 1
    volatility = portfolio_returns.std() * np.sqrt(252)
    
    correlation_matrix = returns.corr()

    return returns, annualized_return, annualized_volatility, cumulative_return, volatility, correlation_matrix, market_returns, portfolio_returns

def calculate_sharpe_ratio(portfolio_return, portfolio_volatility, risk_free_rate):
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
    return sharpe_ratio

def calculate_sortino_ratio(portfolio_returns, risk_free_rate):
    downside_risk = np.sqrt(np.mean(np.minimum(0, portfolio_returns - risk_free_rate / 252) ** 2) * 252)
    portfolio_return = portfolio_returns.mean() * 252
    sortino_ratio = (portfolio_return - risk_free_rate) / downside_risk
    return sortino_ratio

def calculate_treynor_ratio(portfolio_returns, market_returns, risk_free_rate):
    portfolio_return = portfolio_returns.mean() * 252
    market_return = market_returns.mean() * 252
    beta = np.cov(portfolio_returns, market_returns)[0, 1] / np.var(market_returns)
    treynor_ratio = (portfolio_return - risk_free_rate) / beta
    return treynor_ratio

def optimize_portfolio(returns, risk_free_rate):
    def objective(weights):
        portfolio_returns = returns.dot(weights)
        portfolio_return = portfolio_returns.mean() * 252
        portfolio_volatility = portfolio_returns.std() * np.sqrt(252)
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
        return -sharpe_ratio

    num_assets = returns.shape[1]
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
    bounds = [(0, 1) for _ in range(num_assets)]
    initial_weights = np.array(num_assets * [1. / num_assets])

    result = minimize(objective, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)

    return result.x

def plot_portfolio_data(portfolio_return, portfolio_volatility, cumulative_return, correlation_matrix):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    axes[0].bar(["Rentabilidad Anualizada", "Volatilidad Anualizada"], [portfolio_return * 100, portfolio_volatility * 100], color=['blue', 'orange'])
    axes[0].set_title("Rentabilidad y Volatilidad")
    axes[0].set_ylabel('Porcentaje')

    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt='.2f', ax=axes[1])
    axes[1].set_title("Matriz de Correlación")

    st.pyplot(fig)

    st.write(f"Rentabilidad Acumulada: {cumulative_return * 100:.2f}%")
    st.write(f"Volatilidad Anualizada: {portfolio_volatility * 100:.2f}%")

def plot_cml_sml(portfolio_return, portfolio_volatility, market_returns, risk_free_rate):
    market_return = market_returns.mean() * 252
    market_volatility = market_returns.std() * np.sqrt(252)

    volatilities = np.linspace(0, market_volatility * 2, 100)
    returns_cml = risk_free_rate + (portfolio_return - risk_free_rate) / portfolio_volatility * volatilities

    returns_sml = np.linspace(risk_free_rate, market_return * 1.5, 100)
    volatilities_sml = (returns_sml - risk_free_rate) / market_return * market_volatility

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(volatilities, returns_cml, label='Capital Market Line (CML)', color='blue')
    ax.plot(volatilities_sml, returns_sml, label='Security Market Line (SML)', color='red')
    ax.scatter(portfolio_volatility, portfolio_return, color='green', marker='o', s=100, label='Cartera')
    ax.scatter(market_volatility, market_return, color='orange', marker='x', s=100, label='Mercado')

    ax.set_xlabel('Volatilidad')
    ax.set_ylabel('Retorno')
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

def check_normality(returns):
    fig, ax = plt.subplots(figsize=(12, 6))
    
    sns.histplot(returns.mean(axis=1) * 252, kde=True, stat='density', linewidth=0, bins=50, ax=ax)
    
    mu, std = norm.fit(returns.mean(axis=1) * 252)
    xmin, xmax = ax.get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    ax.plot(x, p, 'k', linewidth=2)
    
    ax.set_title("Distribución de los Retornos Anualizados")
    ax.set_xlabel('Retorno Anualizado')
    ax.set_ylabel('Densidad')
    ax.grid(True)
    st.pyplot(fig)

# Interfaz de usuario en Streamlit
st.title('Análisis de Carteras')

tickers, weights, risk_free_rate = get_user_input()

if tickers and weights is not None and risk_free_rate is not None:
    returns, portfolio_return, portfolio_volatility, cumulative_return, volatility, correlation_matrix, market_returns, portfolio_returns = calculate_portfolio_metrics(tickers, weights)

    if returns is not None:
        plot_portfolio_data(portfolio_return, portfolio_volatility, cumulative_return, correlation_matrix)

        sharpe_ratio = calculate_sharpe_ratio(portfolio_return, portfolio_volatility, risk_free_rate)
        st.write(f"Ratio de Sharpe: {sharpe_ratio:.2f}")

        sortino_ratio = calculate_sortino_ratio(portfolio_returns, risk_free_rate)
        st.write(f"Ratio de Sortino: {sortino_ratio:.2f}")

        treynor_ratio = calculate_treynor_ratio(portfolio_returns, market_returns, risk_free_rate)
        st.write(f"Ratio de Treynor: {treynor_ratio:.2f}")

        optimal_weights = optimize_portfolio(returns[tickers], risk_free_rate)

        optimal_portfolio_returns = returns[tickers].dot(optimal_weights)
        optimal_return = optimal_portfolio_returns.mean() * 252
        optimal_volatility = optimal_portfolio_returns.std() * np.sqrt(252)
        optimal_cumulative_return = (1 + optimal_portfolio_returns).prod() - 1

        optimal_sharpe_ratio = calculate_sharpe_ratio(optimal_return, optimal_volatility, risk_free_rate)
        optimal_sortino_ratio = calculate_sortino_ratio(optimal_portfolio_returns, risk_free_rate)
        optimal_treynor_ratio = calculate_treynor_ratio(optimal_portfolio_returns, market_returns, risk_free_rate)

        st.write("\nComposición óptima de la cartera:")
        for ticker, weight in zip(tickers, optimal_weights):
            st.write(f"{ticker}: {weight:.2%}")

        st.write(f"\nRentabilidad media anualizada de la cartera óptima: {optimal_return * 100:.2f}%")
        st.write(f"Volatilidad anualizada de la cartera óptima: {optimal_volatility * 100:.2f}%")
        st.write(f"Rentabilidad Acumulada: {optimal_cumulative_return * 100:.2f}%")
        st.write(f"Ratio de Sharpe de la cartera óptima: {optimal_sharpe_ratio:.2f}")
        st.write(f"Ratio de Sortino de la cartera óptima: {optimal_sortino_ratio:.2f}")
        st.write(f"Ratio de Treynor de la cartera óptima: {optimal_treynor_ratio:.2f}")

        plot_cml_sml(portfolio_return, portfolio_volatility, market_returns, risk_free_rate)
        check_normality(returns[tickers])
