# Import statements

import pandas as pd
import openbb as obb
import matplotlib
import matplotlib.pyplot as plt

# This will obtain daily price data for the complete history of SPY. The start date only needs to be before the first trading day of SPY, no end date is required to obtain the full history.

df = obb.stocks.load("SPY", start_date = '1990-01-01')
obb.ta.standard_deviation
df.head(1)

rvol = obb.ta.rvol_std(df, window = 1).rename("Trading Period 252")
obb.qa.line(
    data= rvol,
    title = "Rolling 30-Day Realized Vol - STD Model",
    log_y = False
)


rvol = pd.concat([rvol, obb.ta.rvol_std(df, window = 2, trading_periods = 365).rename("Trading Perdiod 365")], axis = 1)

obb.forecast.plot(data = rvol, columns = rvol.columns)

rvol = pd.DataFrame()
rvol["STD Model"] = obb.ta.standard_deviation(df)
rvol["Parkinson"] = obb.ta.rvol_parkinson(df)
rvol["Hodges-Tompkins"] = obb.ta.rvol_hodges_tompkins(df)
rvol["Garman-Klass"] = obb.ta.rvol_garman_klass(df)
rvol["Rogers-Satchell"] = obb.ta.rvol_rogers_satchell(df)
rvol["Yang-Zhang"] = obb.ta.rvol_yang_zhang(df)

obb.forecast.plot(data = rvol["2021-06-01":], columns = rvol.columns)

opbb.ta.cones(df)

obb.ta.cones_chart(df, symbol = "SPY")

obb.ta.cones_chart(df, model = 'Yang-Zhang', symbol = 'SPY')

obb.ta.cones_chart(
    data = openbb.stocks.load('SPY'),
    symbol = 'SPY',
    model = 'Garman-Klass',
    upper_q = 0.90,
    lower_q = 0.10
)

from typing import Optional

def rvol(data, window, trading_periods, is_crypto) -> pd.DataFrame:
    
    results = pd.DataFrame()

    results['Standard Deviation'] = obb.ta.rvol_std(data, window, trading_periods, is_crypto)
    results['Parkinson'] = obb.ta.rvol_parkinson(data, window, trading_periods, is_crypto)
    results['Hodges-Tompkins'] = obb.ta.rvol_hodges_tompkins(data, window, trading_periods, is_crypto)
    results['Garman-Klass'] = obb.ta.rvol_garman_klass(data, window, trading_periods, is_crypto)
    results['Rogers-Satchell'] = obb.ta.rvol_rogers_satchell(data, window, trading_periods, is_crypto)
    results['Yang-Zhang'] = obb.ta.rvol_yang_zhang(data, window, trading_periods, is_crypto)
    
    return results


def realized_vol(symbol, window:Optional[int] = 30, trading_periods:Optional[int] = None, is_crypto:Optional[bool] = False) -> pd.DataFrame:
    
    rvol_df = rvol(obb.stocks.load(f"{symbol}"), window, trading_periods, is_crypto)
    
    return rvol_df

data = realized_vol("SPY")

data

data["VIX"] = obb.stocks.load("^VIX")["Adj Close"].rename("VIX")/100

fig = obb.forecast.plot(data, columns = data.columns, external_axes = True)

fig.update_layout(
    {
    'title': 'SPY - 30-Day Realized vs. Implied Volatility',
    'title_y':0.95,
    'title_x':0.5,
    },
    legend=dict(
    yanchor="top",
    y=1,
    xanchor="right",
    orientation="h",
))

%%capture
fig = openbb.forecast.nbeats_chart(
    data = data, 
    target_column = "VIX",
    past_covariates = "Standard Deviation,Parkinson,Hodges-Tompkins,Rogers-Satchell,Yang-Zhang",
    forecast_only = True,
    external_axes = True
)

fig.show()
