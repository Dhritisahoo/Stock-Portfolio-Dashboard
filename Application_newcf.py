#!/usr/bin/env python
# coding: utf-8

# In[4]:


import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import yfinance as yf
import pandas as pd
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import panel as pn

pn.extension('plotly', 'tabulator')

# Select your fund tickers
tickers = ["OIEIX", "JLGRX", "JGASX", "FBGRX", "PEMPX", "GSCGX", "GGOIX", "VEIPX", "GPIRX", "VIVIX"]

def create_db(tickers):
    # Creating an empty DataFrame to store the fund data
    df = pd.DataFrame()

    for ticker in tickers:
        # Download the historical data from Yahoo Finance
        fund_data = yf.download(ticker, start="2021-01-01", end="2023-07-31")

        # Extract the adjusted close prices
        close_prices = fund_data["Adj Close"]

        # Rename the column with the fund ticker
        close_prices = close_prices.rename(ticker)

        # Append the fund's close prices to the DataFrame
        df = pd.concat([df, close_prices], axis=1)

    return df

# Create the DataFrame with all the funds
fund_data = create_db(tickers)



# In[6]:


def calculate_sharpe_ratio(returns, risk_free_rate):
    excess_returns = returns - risk_free_rate
    sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns)
    return sharpe_ratio

def calculate_growth(principal, returns):
    compounded_growth = np.prod(1 + returns)
    final_value = principal * compounded_growth
    return final_value


# In[25]:


import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import yfinance as yf
import pandas as pd
import numpy as np

# Select your fund tickers
tickers = ["OIEIX", "JLGRX", "JGASX", "FBGRX", "PEMPX", "GSCGX", "GGOIX", "VEIPX", "GPIRX", "VIVIX"]

def create_db(tickers):
    # Creating an empty DataFrame to store the fund data
    df = pd.DataFrame()

    for ticker in tickers:
        # Download the historical data from Yahoo Finance
        fund_data = yf.download(ticker, start="2021-01-01", end="2023-07-31")

        # Extract the adjusted close prices
        close_prices = fund_data["Adj Close"]

        # Rename the column with the fund ticker
        close_prices = close_prices.rename(ticker)

        # Append the fund's close prices to the DataFrame
        df = pd.concat([df, close_prices], axis=1)

    return df

# Create the DataFrame with all the funds
fund_data = create_db(tickers)

def calculate_sharpe_ratio(returns, risk_free_rate):
    excess_returns = returns - risk_free_rate
    sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns)
    return sharpe_ratio

def calculate_growth(principal, returns):
    compounded_growth = np.prod(1 + returns)
    final_value = principal * compounded_growth
    return final_value

# Create the Dash app
app = dash.Dash(__name__)

# Define the dropdown options for stocks
dropdown_options = [{'label': ticker, 'value': ticker} for ticker in tickers]

# Create the layout of the app
app.layout = html.Div([
    html.H1("Stock Portfolio Dashboard"),
    html.Div([
        html.Label("Select a stock:"),
        dcc.Dropdown(
            id="stock-dropdown",
            options=dropdown_options,
            value=tickers[0]
        ),
        html.Label("Enter weights for each investor type:"),
        html.Div([
            html.Label("Conservative:"),
            dcc.Input(
                id="conservative-input",
                type="number",
                value=0,
                min=0,
                max=100,
                step=1
            )
        ]),
        html.Div([
            html.Label("Moderate:"),
            dcc.Input(
                id="moderate-input",
                type="number",
                value=0,
                min=0,
                max=100,
                step=1
            )
        ]),
        html.Div([
            html.Label("Aggressive:"),
            dcc.Input(
                id="aggressive-input",
                type="number",
                value=0,
                min=0,
                max=100,
                step=1
            )
        ]),
        html.Button("Add Stock", id="add-button", n_clicks=0),
        html.Button("Calculate", id="calculate-button", n_clicks=0)
    ]),
    html.Div(id="portfolio-table"),
    html.Div(id="portfolio-metrics")
])

# Callback to update the portfolio table
@app.callback(
    Output("portfolio-table", "children"),
    [Input("add-button", "n_clicks")],
    [State("stock-dropdown", "value"),
     State("conservative-input", "value"),
     State("moderate-input", "value"),
     State("aggressive-input", "value"),
     State("portfolio-table", "children")]
)
def update_portfolio_table(add_clicks, stock_value, conservative_weight, moderate_weight, aggressive_weight, existing_table):
    if existing_table is None:
        existing_table = []  # Initialize as an empty list if None

    if add_clicks > 0:
        existing_table.append(html.Tr([
            html.Td(stock_value),
            html.Td(f"Conservative: {conservative_weight}%, Moderate: {moderate_weight}%, Aggressive: {aggressive_weight}%")
        ]))
        return existing_table
    else:
        return existing_table

# Callback to calculate portfolio returns, risk, maximum drawdown, Sharpe ratio, and growth
@app.callback(
    Output("portfolio-metrics", "children"),
    [Input("calculate-button", "n_clicks")],
    [State("portfolio-table", "children")]
)
def calculate_portfolio_metrics_callback(calculate_clicks, existing_table):
    if calculate_clicks > 0:
        if len(existing_table) == 0:
            return html.Div("No stocks added.", style={'color': 'red'})

        stocks = []
        conservative_weights = []
        moderate_weights = []
        aggressive_weights = []

        for row in existing_table:
            cells = row["props"]["children"]
            stock = cells[0]["props"]["children"]
            weights_str = cells[1]["props"]["children"].split(", ")
            conservative_weight = float(weights_str[0].split(": ")[1].split("%")[0]) / 100.0
            moderate_weight = float(weights_str[1].split(": ")[1].split("%")[0]) / 100.0
            aggressive_weight = float(weights_str[2].split(": ")[1].split("%")[0]) / 100.0

            stocks.append(stock)
            conservative_weights.append(conservative_weight)
            moderate_weights.append(moderate_weight)
            aggressive_weights.append(aggressive_weight)

        selected_stocks = fund_data[stocks]
        returns = selected_stocks.pct_change()

        conservative_portfolio_returns = returns.dot(conservative_weights)
        moderate_portfolio_returns = returns.dot(moderate_weights)
        aggressive_portfolio_returns = returns.dot(aggressive_weights)

        def calculate_monthly_max_drawdown(returns):
            cum_returns = (1 + returns).cumprod()
            rolling_max = cum_returns.expanding().max()
            drawdown = (cum_returns / rolling_max) - 1
            monthly_max_drawdown = drawdown.groupby(drawdown.index.to_period('M')).min()
            return monthly_max_drawdown * 100

        conservative_monthly_max_drawdown = calculate_monthly_max_drawdown(conservative_portfolio_returns)
        moderate_monthly_max_drawdown = calculate_monthly_max_drawdown(moderate_portfolio_returns)
        aggressive_monthly_max_drawdown = calculate_monthly_max_drawdown(aggressive_portfolio_returns)

        risk_free_rate = 0.02  # Set your risk-free rate here

        annualization_factor = 252  # Assuming daily returns, change if using other frequency

        conservative_portfolio_sharpe_ratio = calculate_sharpe_ratio(conservative_portfolio_returns * annualization_factor, risk_free_rate)
        moderate_portfolio_sharpe_ratio = calculate_sharpe_ratio(moderate_portfolio_returns * annualization_factor, risk_free_rate)
        aggressive_portfolio_sharpe_ratio = calculate_sharpe_ratio(aggressive_portfolio_returns * annualization_factor, risk_free_rate)

        principal = 1000000  # $1 million

        conservative_portfolio_growth = calculate_growth(principal, conservative_portfolio_returns)
        moderate_portfolio_growth = calculate_growth(principal, moderate_portfolio_returns)
        aggressive_portfolio_growth = calculate_growth(principal, aggressive_portfolio_returns)

        return html.Div([
            html.Div("Conservative Investor:", style={'font-weight': 'bold'}),
            html.Div(f"Portfolio Return: {conservative_portfolio_returns.mean() * annualization_factor:.4f}", style={'margin-bottom': '10px'}),
            html.Div(f"Portfolio Risk: {conservative_portfolio_returns.std() * np.sqrt(annualization_factor):.4f}", style={'margin-bottom': '10px'}),
            html.Div(f"Monthly Maximum Drawdown: {conservative_monthly_max_drawdown.min():.2f}%", style={'margin-bottom': '10px'}),
            html.Div(f"Sharpe Ratio: {conservative_portfolio_sharpe_ratio:.4f}", style={'margin-bottom': '10px'}),
            html.Div(f"Growth: {conservative_portfolio_growth:.2f}", style={'margin-bottom': '20px'}),
            html.Div("Moderate Investor:", style={'font-weight': 'bold'}),
            html.Div(f"Portfolio Return: {moderate_portfolio_returns.mean() * annualization_factor:.4f}", style={'margin-bottom': '10px'}),
            html.Div(f"Portfolio Risk: {moderate_portfolio_returns.std() * np.sqrt(annualization_factor):.4f}", style={'margin-bottom': '10px'}),
            html.Div(f"Monthly Maximum Drawdown: {moderate_monthly_max_drawdown.min():.2f}%", style={'margin-bottom': '10px'}),
            html.Div(f"Sharpe Ratio: {moderate_portfolio_sharpe_ratio:.4f}", style={'margin-bottom': '10px'}),
            html.Div(f"Growth: {moderate_portfolio_growth:.2f}", style={'margin-bottom': '20px'}),
            html.Div("Aggressive Investor:", style={'font-weight': 'bold'}),
            html.Div(f"Portfolio Return: {aggressive_portfolio_returns.mean() * annualization_factor:.4f}", style={'margin-bottom': '10px'}),
            html.Div(f"Portfolio Risk: {aggressive_portfolio_returns.std() * np.sqrt(annualization_factor):.4f}", style={'margin-bottom': '10px'}),
            html.Div(f"Monthly Maximum Drawdown: {aggressive_monthly_max_drawdown.min():.2f}%", style={'margin-bottom': '10px'}),
            html.Div(f"Sharpe Ratio: {aggressive_portfolio_sharpe_ratio:.4f}", style={'margin-bottom': '10px'}),
            html.Div(f"Growth: {aggressive_portfolio_growth:.2f}", style={'margin-bottom': '20px'})
        ])
    else:
        return html.Div()

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True, port=8068)


# In[10]:




