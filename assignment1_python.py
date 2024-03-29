import pandas as pd
from numpy.linalg import solve
import numpy as np

# Load the dataset
df = pd.read_csv('current.csv')

# Clean the DataFrame by removing the row with transformation codes
df_cleaned = df.drop(index=0)
df_cleaned.reset_index(drop=True, inplace=True)

## df_cleaned contains the data cleaned
df_cleaned

# Extract transformation codes
transformation_codes = df.iloc[0, 1:].to_frame().reset_index()
transformation_codes.columns = ['Series', 'Transformation_Code']

## transformation_codes contains the transformation codes
## - `transformation_code=1`: no trasformation
## - `transformation_code=2`: $\Delta x_t$
## - `transformation_code=3`: $\Delta^2 x_t$
## - `transformation_code=4`: $log(x_t)$
## - `transformation_code=5`: $\Delta log(x_t)$
## - `transformation_code=6`: $\Delta^2 log(x_t)$
## - `transformation_code=7`: $\Delta (x_t/x_{t-1} - 1)$



# Function to apply transformations based on the transformation code
def apply_transformation(series, code):
    if code == 1:
        # No transformation
        return series
    elif code == 2:
        # First difference
        return series.diff()
    elif code == 3:
        # Second difference
        return series.diff().diff()
    elif code == 4:
        # Log
        return np.log(series)
    elif code == 5:
        # First difference of log
        return np.log(series).diff()
    elif code == 6:
        # Second difference of log
        return np.log(series).diff().diff()
    elif code == 7:
        # Delta (x_t/x_{t-1} - 1)
        return series.pct_change()
    else:
        raise ValueError("Invalid transformation code")

# Applying the transformations to each column in df_cleaned based on transformation_codes
for series_name, code in transformation_codes.values:
    df_cleaned[series_name] = apply_transformation(df_cleaned[series_name].astype(float), float(code))

df_cleaned.head()

## Plot the transformed series
series_to_plot = ['INDPRO', 'CPIAUCSL', 'TB3MS']
series_names = ['Industrial Production', 'Inflation (CPI)', 'Federal Funds Rate']
# 'INDPRO'   for Industrial Production, 
# 'CPIAUCSL' for Inflation (Consumer Price Index), 
# 'TB3MS'    3-month treasury bill.
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Create a figure and a grid of subplots
fig, axs = plt.subplots(len(series_to_plot), 1, figsize=(10, 15))

# Iterate over the selected series and plot each one
for ax, series_name, plot_title in zip(axs, series_to_plot, series_names):
    if series_name in df_cleaned.columns:
        # Convert 'sasdate' to datetime format for plotting
        dates = pd.to_datetime(df_cleaned['sasdate'], format='%m/%d/%Y')
        ax.plot(dates, df_cleaned[series_name], label=plot_title)
        # Formatting the x-axis to show only every five years
        ax.xaxis.set_major_locator(mdates.YearLocator(base=5))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.set_title(plot_title)
        ax.set_xlabel('Year')
        ax.set_ylabel('Transformed Value')
        ax.legend(loc='upper left')
        # Improve layout of date labels
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    else:
        ax.set_visible(False)  # Hide plots for which the data is not available

plt.tight_layout()
plt.show()

Y = df_cleaned['INDPRO'].dropna()
X = df_cleaned[['CPIAUCSL', 'FEDFUNDS']].dropna()

h = 1 ## One-step ahead
p = 4
r = 4

Y_target = Y.shift(-h).dropna()
Y_lagged = pd.concat([Y.shift(i) for i in range(p+1)], axis=1).dropna()
X_lagged = pd.concat([X.shift(i) for i in range(r+1)], axis=1).dropna()
common_index = Y_lagged.index.intersection(Y_target.index)
common_index = common_index.intersection(X_lagged.index)

## This is the last row needed to create the forecast
X_T = np.concatenate([[1], Y_lagged.iloc[-1], X_lagged.iloc[-1]])

## Align the data
Y_target = Y_target.loc[common_index]
Y_lagged = Y_lagged.loc[common_index]
X_lagged = X_lagged.loc[common_index]

X_reg = pd.concat([X_lagged, Y_lagged], axis = 1)



X_reg = pd.concat([X_lagged, Y_lagged], axis=1)
X_reg_np = np.concatenate([np.ones((X_reg.shape[0], 1)), X_reg.values], axis=1)
Y_target_np = Y_target.values

# Solving for the OLS estimator beta: (X'X)^{-1} X'Y
beta_ols = solve(X_reg_np.T @ X_reg_np, X_reg_np.T @ Y_target_np)

## Produce the One step ahead forecast
## % change month-to-month INDPRO
print(X_T)
forecast = X_T@beta_ols*100
print("Forecast for Industrial production:", forecast)
print("Model parameter estimates for Industrial production:", beta_ols)

# Select variables for modeling
Y_cpi = df_cleaned['CPIAUCSL'].dropna()
X_cpi = df_cleaned[['INDPRO', 'FEDFUNDS']].dropna()

# Define model parameters
p = 4  # number of lags for the INDPRO variable
r = 4  # number of lags for the FEDFUNDS variable
h = 1  # one-step ahead forecast

# Prepare data using lagged variables
Y_cpi_target = Y_cpi.shift(-h).dropna()
Y_cpi_lagged = pd.concat([Y_cpi.shift(i) for i in range(p+1)], axis=1).dropna()
X_cpi_lagged = pd.concat([X_cpi.shift(i) for i in range(r+1)], axis=1).dropna()
common_index_cpi = Y_cpi_lagged.index.intersection(Y_cpi_target.index)
common_index_cpi = common_index_cpi.intersection(X_cpi_lagged.index)

# Get the last row of data to create the forecast
X_cpi_T = np.concatenate([[1], Y_cpi_lagged.iloc[-1], X_cpi_lagged.iloc[-1]])

# Select common data for modeling
Y_cpi_target = Y_cpi_target.loc[common_index_cpi]
Y_cpi_lagged = Y_cpi_lagged.loc[common_index_cpi]
X_cpi_lagged = X_cpi_lagged.loc[common_index_cpi]

# Gather data for regression
X_cpi_reg = pd.concat([X_cpi_lagged, Y_cpi_lagged], axis=1)
X_cpi_reg_np = np.concatenate([np.ones((X_cpi_reg.shape[0], 1)), X_cpi_reg.values], axis=1)
Y_cpi_target_np = Y_cpi_target.values

# Estimate parameters using OLS regression
beta_ols_cpi = solve(X_cpi_reg_np.T @ X_cpi_reg_np, X_cpi_reg_np.T @ Y_cpi_target_np)

# Make a one-step ahead forecast
forecast_cpi = X_cpi_T @ beta_ols_cpi * 100
print("Forecast for Inflation:", forecast_cpi)
print("Model parameter estimates for Inflation:", beta_ols_cpi)


# Select variables for modeling
Y_tb3ms = df_cleaned['TB3MS'].dropna()
X_tb3ms = df_cleaned[['INDPRO', 'CPIAUCSL']].dropna()

# Define model parameters
p = 4  # number of lags for the INDPRO variable
r = 4  # number of lags for the CPIAUCSL variable
h = 1  # one-step ahead forecast

# Prepare data using lagged variables
Y_tb3ms_target = Y_tb3ms.shift(-h).dropna()
Y_tb3ms_lagged = pd.concat([Y_tb3ms.shift(i) for i in range(p+1)], axis=1).dropna()
X_tb3ms_lagged = pd.concat([X_tb3ms.shift(i) for i in range(r+1)], axis=1).dropna()
common_index_tb3ms = Y_tb3ms_lagged.index.intersection(Y_tb3ms_target.index)
common_index_tb3ms = common_index_tb3ms.intersection(X_tb3ms_lagged.index)

# Get the last row of data to create the forecast
X_tb3ms_T = np.concatenate([[1], Y_tb3ms_lagged.iloc[-1], X_tb3ms_lagged.iloc[-1]])

# Select common data for modeling
Y_tb3ms_target = Y_tb3ms_target.loc[common_index_tb3ms]
Y_tb3ms_lagged = Y_tb3ms_lagged.loc[common_index_tb3ms]
X_tb3ms_lagged = X_tb3ms_lagged.loc[common_index_tb3ms]

# Gather data for regression
X_tb3ms_reg = pd.concat([X_tb3ms_lagged, Y_tb3ms_lagged], axis=1)
X_tb3ms_reg_np = np.concatenate([np.ones((X_tb3ms_reg.shape[0], 1)), X_tb3ms_reg.values], axis=1)
Y_tb3ms_target_np = Y_tb3ms_target.values

# Estimate parameters using OLS regression
beta_ols_tb3ms = solve(X_tb3ms_reg_np.T @ X_tb3ms_reg_np, X_tb3ms_reg_np.T @ Y_tb3ms_target_np)

# Make a one-step ahead forecast
forecast_tb3ms = X_tb3ms_T @ beta_ols_tb3ms * 100
print("Forecast for Interest rates:", forecast_tb3ms)
print("Model parameter estimates for Interest rates:", beta_ols_tb3ms)

def MSFE(true_values, forecast_values):
    return ((true_values - forecast_values) ** 2).mean()

# Define the size of the expanding window
window_size = 60

# Initialize lists to store MSFE values
msfe_values_indpro = []
msfe_values_cpi = []
msfe_values_tb3ms = []

# Iterate over the data using an expanding window
for i in range(len(df_cleaned) - window_size + 1):
    # Define the current expanding window
    df_window = df_cleaned.iloc[:i + window_size]
    
    # Prepare data for forecasting the INDPRO variable
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

    # Prepare data for forecasting the CPIAUCSL variable
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

    # Prepare data for forecasting the TB3MS variable
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

