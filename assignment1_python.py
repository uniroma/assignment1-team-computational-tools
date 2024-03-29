import pandas as pd
from numpy.linalg import solve
import numpy as np

df = pd.read_csv('current.csv')

df_cleaned = df.drop(index=0)
df_cleaned.reset_index(drop=True, inplace=True)

df_cleaned

transformation_codes = df.iloc[0, 1:].to_frame().reset_index()
transformation_codes.columns = ['Series', 'Transformation_Code']

def apply_transformation(series, code):
    if code == 1:
        return series
    elif code == 2:
        return series.diff()
    elif code == 3:
        return series.diff().diff()
    elif code == 4:
        return np.log(series)
    elif code == 5:
        return np.log(series).diff()
    elif code == 6:
        return np.log(series).diff().diff()
    elif code == 7:
        return series.pct_change()
    else:
        raise ValueError("Invalid transformation code")

for series_name, code in transformation_codes.values:
    df_cleaned[series_name] = apply_transformation(df_cleaned[series_name].astype(float), float(code))

df_cleaned.head()

series_to_plot = ['INDPRO', 'CPIAUCSL', 'TB3MS']
series_names = ['Industrial Production', 'Inflation (CPI)', 'Federal Funds Rate']

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

fig, axs = plt.subplots(len(series_to_plot), 1, figsize=(10, 15))

for ax, series_name, plot_title in zip(axs, series_to_plot, series_names):
    if series_name in df_cleaned.columns:
        dates = pd.to_datetime(df_cleaned['sasdate'], format='%m/%d/%Y')
        ax.plot(dates, df_cleaned[series_name], label=plot_title)
        ax.xaxis.set_major_locator(mdates.YearLocator(base=5))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.set_title(plot_title)
        ax.set_xlabel('Year')
        ax.set_ylabel('Transformed Value')
        ax.legend(loc='upper left')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    else:
        ax.set_visible(False)  
plt.tight_layout()
plt.show()

Y = df_cleaned['INDPRO'].dropna()
X = df_cleaned[['CPIAUCSL', 'FEDFUNDS']].dropna()

h = 1 
p = 4
r = 4

Y_target = Y.shift(-h).dropna()
Y_lagged = pd.concat([Y.shift(i) for i in range(p+1)], axis=1).dropna()
X_lagged = pd.concat([X.shift(i) for i in range(r+1)], axis=1).dropna()
common_index = Y_lagged.index.intersection(Y_target.index)
common_index = common_index.intersection(X_lagged.index)

X_T = np.concatenate([[1], Y_lagged.iloc[-1], X_lagged.iloc[-1]])

Y_target = Y_target.loc[common_index]
Y_lagged = Y_lagged.loc[common_index]
X_lagged = X_lagged.loc[common_index]

X_reg = pd.concat([X_lagged, Y_lagged], axis = 1)



X_reg = pd.concat([X_lagged, Y_lagged], axis=1)
X_reg_np = np.concatenate([np.ones((X_reg.shape[0], 1)), X_reg.values], axis=1)
Y_target_np = Y_target.values

beta_ols = solve(X_reg_np.T @ X_reg_np, X_reg_np.T @ Y_target_np)

print(X_T)
forecast = X_T@beta_ols*100
print("Forecast for Industrial production:", forecast)
print("Model parameter estimates for Industrial production:", beta_ols)

Y_cpi = df_cleaned['CPIAUCSL'].dropna()
X_cpi = df_cleaned[['INDPRO', 'FEDFUNDS']].dropna()

p = 4  
r = 4  
h = 1  

Y_cpi_target = Y_cpi.shift(-h).dropna()
Y_cpi_lagged = pd.concat([Y_cpi.shift(i) for i in range(p+1)], axis=1).dropna()
X_cpi_lagged = pd.concat([X_cpi.shift(i) for i in range(r+1)], axis=1).dropna()
common_index_cpi = Y_cpi_lagged.index.intersection(Y_cpi_target.index)
common_index_cpi = common_index_cpi.intersection(X_cpi_lagged.index)

X_cpi_T = np.concatenate([[1], Y_cpi_lagged.iloc[-1], X_cpi_lagged.iloc[-1]])

Y_cpi_target = Y_cpi_target.loc[common_index_cpi]
Y_cpi_lagged = Y_cpi_lagged.loc[common_index_cpi]
X_cpi_lagged = X_cpi_lagged.loc[common_index_cpi]

X_cpi_reg = pd.concat([X_cpi_lagged, Y_cpi_lagged], axis=1)
X_cpi_reg_np = np.concatenate([np.ones((X_cpi_reg.shape[0], 1)), X_cpi_reg.values], axis=1)
Y_cpi_target_np = Y_cpi_target.values

beta_ols_cpi = solve(X_cpi_reg_np.T @ X_cpi_reg_np, X_cpi_reg_np.T @ Y_cpi_target_np)

forecast_cpi = X_cpi_T @ beta_ols_cpi * 100
print("Forecast for Inflation:", forecast_cpi)
print("Model parameter estimates for Inflation:", beta_ols_cpi)

Y_tb3ms = df_cleaned['TB3MS'].dropna()
X_tb3ms = df_cleaned[['INDPRO', 'CPIAUCSL']].dropna()

p = 4  
r = 4  
h = 1  

Y_tb3ms_target = Y_tb3ms.shift(-h).dropna()
Y_tb3ms_lagged = pd.concat([Y_tb3ms.shift(i) for i in range(p+1)], axis=1).dropna()
X_tb3ms_lagged = pd.concat([X_tb3ms.shift(i) for i in range(r+1)], axis=1).dropna()
common_index_tb3ms = Y_tb3ms_lagged.index.intersection(Y_tb3ms_target.index)
common_index_tb3ms = common_index_tb3ms.intersection(X_tb3ms_lagged.index)

X_tb3ms_T = np.concatenate([[1], Y_tb3ms_lagged.iloc[-1], X_tb3ms_lagged.iloc[-1]])

Y_tb3ms_target = Y_tb3ms_target.loc[common_index_tb3ms]
Y_tb3ms_lagged = Y_tb3ms_lagged.loc[common_index_tb3ms]
X_tb3ms_lagged = X_tb3ms_lagged.loc[common_index_tb3ms]

X_tb3ms_reg = pd.concat([X_tb3ms_lagged, Y_tb3ms_lagged], axis=1)
X_tb3ms_reg_np = np.concatenate([np.ones((X_tb3ms_reg.shape[0], 1)), X_tb3ms_reg.values], axis=1)
Y_tb3ms_target_np = Y_tb3ms_target.values

beta_ols_tb3ms = solve(X_tb3ms_reg_np.T @ X_tb3ms_reg_np, X_tb3ms_reg_np.T @ Y_tb3ms_target_np)

forecast_tb3ms = X_tb3ms_T @ beta_ols_tb3ms * 100
print("Forecast for Interest rates:", forecast_tb3ms)
print("Model parameter estimates for Interest rates:", beta_ols_tb3ms)

def MSFE(true_values, forecast_values):
    return ((true_values - forecast_values) ** 2).mean()

window_size = 60

msfe_values_indpro = []
msfe_values_cpi = []
msfe_values_tb3ms = []

for i in range(len(df_cleaned) - window_size + 1):
    df_window = df_cleaned.iloc[:i + window_size]
    
    Y_indpro = df_window['INDPRO']
    X_indpro = df_window[['CPIAUCSL', 'TB3MS']]
    p = 4
    r = 4
    h = 1
    Y_indpro_target = Y_indpro.shift(-h).dropna()
    Y_indpro_lagged = pd.concat([Y_indpro.shift(i) for i in range(p+1)], axis=1).dropna()
    X_indpro_lagged = pd.concat([X_indpro.shift(i) for i in range(r+1)], axis=1).dropna()
    common_index_indpro = Y_indpro_lagged.index.intersection(Y_indpro_target.index)
    common_index_indpro = common_index_indpro.intersection(X_indpro_lagged.index)
    X_indpro_T = np.concatenate([[1], Y_indpro_lagged.iloc[-1], X_indpro_lagged.iloc[-1]])
    Y_indpro_target = Y_indpro_target.loc[common_index_indpro]
    Y_indpro_lagged = Y_indpro_lagged.loc[common_index_indpro]
    X_indpro_lagged = X_indpro_lagged.loc[common_index_indpro]
    X_indpro_reg = pd.concat([X_indpro_lagged, Y_indpro_lagged], axis = 1)
    X_indpro_reg = pd.concat([X_indpro_lagged, Y_indpro_lagged], axis=1)
    X_indpro_reg_np = np.concatenate([np.ones((X_indpro_reg.shape[0], 1)), X_indpro_reg.values], axis=1)
    Y_indpro_target_np = Y_indpro_target.values
    beta_ols_indpro = solve(X_indpro_reg_np.T @ X_indpro_reg_np, X_indpro_reg_np.T @ Y_indpro_target_np)
    forecast_indpro = X_indpro_T @ beta_ols_indpro * 100
    true_values_indpro = df_window['INDPRO'].iloc[-1]
    msfe_indpro = MSFE(true_values_indpro, forecast_indpro)
    msfe_values_indpro.append(msfe_indpro)

    Y_cpi = df_window['CPIAUCSL']
    X_cpi = df_window[['INDPRO', 'FEDFUNDS']]
    p = 4
    r = 4
    h = 1
    Y_cpi_target = Y_cpi.shift(-h).dropna()
    Y_cpi_lagged = pd.concat([Y_cpi.shift(i) for i in range(p+1)], axis=1).dropna()
    X_cpi_lagged = pd.concat([X_cpi.shift(i) for i in range(r+1)], axis=1).dropna()
    common_index_cpi = Y_cpi_lagged.index.intersection(Y_cpi_target.index)
    common_index_cpi = common_index_cpi.intersection(X_cpi_lagged.index)
    X_cpi_T = np.concatenate([[1], Y_cpi_lagged.iloc[-1], X_cpi_lagged.iloc[-1]])
    Y_cpi_target = Y_cpi_target.loc[common_index_cpi]
    Y_cpi_lagged = Y_cpi_lagged.loc[common_index_cpi]
    X_cpi_lagged = X_cpi_lagged.loc[common_index_cpi]
    X_cpi_reg = pd.concat([X_cpi_lagged, Y_cpi_lagged], axis = 1)
    X_cpi_reg = pd.concat([X_cpi_lagged, Y_cpi_lagged], axis=1)
    X_cpi_reg_np = np.concatenate([np.ones((X_cpi_reg.shape[0], 1)), X_cpi_reg.values], axis=1)
    Y_cpi_target_np = Y_cpi_target.values
    beta_ols_cpi = solve(X_cpi_reg_np.T @ X_cpi_reg_np, X_cpi_reg_np.T @ Y_cpi_target_np)
    forecast_cpi = X_cpi_T @ beta_ols_cpi * 100
    true_values_cpi = df_window['CPIAUCSL'].iloc[-1]
    msfe_cpi = MSFE(true_values_cpi, forecast_cpi)
    msfe_values_cpi.append(msfe_cpi)

    Y_tb3ms = df_window['TB3MS']
    X_tb3ms = df_window[['INDPRO', 'CPIAUCSL']]
    p = 4
    r = 4
    h = 1
    Y_tb3ms_target = Y_tb3ms.shift(-h).dropna()
    Y_tb3ms_lagged = pd.concat([Y_tb3ms.shift(i) for i in range(p+1)], axis=1).dropna()
    X_tb3ms_lagged = pd.concat([X_tb3ms.shift(i) for i in range(r+1)], axis=1).dropna()
    common_index_tb3ms = Y_tb3ms_lagged.index.intersection(Y_tb3ms_target.index)
    common_index_tb3ms = common_index_tb3ms.intersection(X_tb3ms_lagged.index)
    X_tb3ms_T = np.concatenate([[1], Y_tb3ms_lagged.iloc[-1], X_tb3ms_lagged.iloc[-1]])
    Y_tb3ms_target = Y_tb3ms_target.loc[common_index_tb3ms]
    Y_tb3ms_lagged = Y_tb3ms_lagged.loc[common_index_tb3ms]
    X_tb3ms_lagged = X_tb3ms_lagged.loc[common_index_tb3ms]
    X_tb3ms_reg = pd.concat([X_tb3ms_lagged, Y_tb3ms_lagged], axis = 1)
    X_tb3ms_reg = pd.concat([X_tb3ms_lagged, Y_tb3ms_lagged], axis=1)
    X_tb3ms_reg_np = np.concatenate([np.ones((X_tb3ms_reg.shape[0], 1)), X_tb3ms_reg.values], axis=1)
    Y_tb3ms_target_np = Y_tb3ms_target.values
    beta_ols_tb3ms = solve(X_tb3ms_reg_np.T @ X_tb3ms_reg_np, X_tb3ms_reg_np.T @ Y_tb3ms_target_np)
    forecast_tb3ms = X_tb3ms_T @ beta_ols_tb3ms * 100
    true_values_tb3ms = df_window['TB3MS'].iloc[-1]
    msfe_tb3ms = MSFE(true_values_tb3ms, forecast_tb3ms)
    msfe_values_tb3ms.append(msfe_tb3ms)

# Print the average MSFE values for each variable
print("Average MSFE for INDPRO:", np.mean(msfe_values_indpro))
print("Average MSFE for CPIAUCSL:", np.mean(msfe_values_cpi))
print("Average MSFE for TB3MS:", np.mean(msfe_values_tb3ms))  

