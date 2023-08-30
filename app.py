# %%
import datetime
import numpy as np
import pandas as pd

# %%
import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output

# %%
import plotly.graph_objs as go
import plotly.io as pio

# %%
from openbb_terminal.sdk import openbb

# %%
from sklearn.decomposition import PCA

# %%
pio.templates.default="plotly"

# %%
app=dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# %%
ticker_field= [
    html.Label("Enter Ticker Symbols:"),
    dcc.Input(
    id="ticker-input",
    type="text",
    ),
]

# %%
components_field=[
    html.Label("Select Number of Components:"),
    dcc.Dropdown(
        id="component-dropdown",
        options=[{"label":i,"value":i} for i in
range(1,6)],
        value=3,
    ),
]

# %%
date_picker_field=[
    html.Label("Select Date Range:"),
    dcc.DatePickerRange(
        id="date-picker",
    start_date=datetime.datetime.now()-
datetime.timedelta(365*3),
    end_date=datetime.datetime.now(),
    display_format="YYY-MM-DD"
    ),
]
submit=[
    html.Button("Submit",id="submit-button"),
]

# %%
app.layout=dbc.Container(
 [
     html.H1("PCS on Stock Returns"),
     #Ticker input
     dbc.Row([dbc.Col(ticker_field)]),
     dbc.Row([dbc.Col(components_field)]),
     dbc.Row([dbc.Col(date_picker_field)]),
     dbc.Row([dbc.Col(submit)]),
     
     #Charts
     dbc.Row(
        [
            dbc.Col([dcc.Graph(id="bar-chart")],
width=4),
            dbc.Col([dcc.Graph(id="line-chart")],
width=4),
        ]  
     ),
    ]
  )

# %%
@app.callback(
    [
        Output("bar-chart","figure"),
        Output("line-chart","figure"),
    ],
    
    [Input("submit-button","n_clicks")],
    
    [
        dash.dependencies.State("ticker-input","value"),
        dash.dependencies.State("component-dropdown","value"),
        dash.dependencies.State("date-picker", "start-date"),
        dash.dependencies.State("date-picker","end_date"),
    ],
 )
def update_graphs (n_clicks, tickers, n_components, start_Date, end_date):
    if not tickers:
        return {}, {}
    #Parse inputs from user
    tickers=tickers.split(",")
    
    start_date=datetime.datetime.strptime(
         start_date,
        "%Y-%m-%dT%H:%M:%S.%f"
    ).date()
    
    #Download stock data
    data=openbb.economy.index(
        tickers,
        start_date=start_date,
        end_date=end_date
    )
    daily_returns=data.pct_change().dropna()
    
    #Apply PCA
    pca= PCA(n_components=n_components)
    pca.fit(daily_returns)
    
    explained_var_ratio=pca.explained_variancee_ratio_
    
    #Bar chart for individual explained variance
    bar_chart=go.Figure(
        data=[
            go.Bar(
            x=["PC" + str(i+1) for i in
range (n_components)],
                  y=explained_var_ratio,
            )
        ],
        layout=go.Layout(
        title="Explained Variance by Component",
        xaxis=dict(title="Principal Component"),
        yaxis=dict(title="Cumulative Explained Variance"),
        ),
    )
    return bar_chart, line_chart

if __name__== "__main__":
    app.run_server(debug=True)
