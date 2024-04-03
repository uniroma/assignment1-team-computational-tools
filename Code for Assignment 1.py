import pandas as pd
import numpy as np
from numpy.linalg import inv

# importing dataframe
df = pd.read_csv('~/Downloads/current.csv')
df_cleaned = df.drop(index=0)
df_cleaned.reset_index(drop=True, inplace=True)
transformation_codes = df.iloc[0, 1:].to_frame().reset_index()
transformation_codes.columns = ['Series', 'Transformation_Code']

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

for series_name, code in transformation_codes.values:
    df_cleaned[series_name] = apply_transformation(df_cleaned[series_name].astype(float), float(code))

df_cleaned['sasdate'] = pd.to_datetime(df_cleaned['sasdate']) # convert sasdate to datetime

#define function to obtain betas and forecasts
def calculate_betas(h=1, p=4,
                    target = 'INDPRO',
                    CoVars = ['CPIAUCSL', 'TB3MS'], end_date = '1/1/2024'):

    new_data = df_cleaned[df_cleaned['sasdate'] <= pd.Timestamp(end_date)] # subset data
    Reg_data = pd.DataFrame() # initiate empty dataset to store all variables for estimation
    Reg_data['Y_lead'] = new_data[target].shift(-h) #lead the target variable
    #lagging Target Variable:
    for i in range(0, p+1):
        Reg_data[f'{target}_lag{i}'] = new_data[target].shift(i)

    #lagging Covariates:
    for col in CoVars:
        for i in range(0, p+1):
            Reg_data[f'{col}_lag{i}'] = new_data[col].shift(i)

    #Drop NA values:
    Reg_data = Reg_data.dropna()
    Y_reg = Reg_data['Y_lead'].values # target vector
    X_reg = Reg_data.iloc[:,1:] # Covariate matrix
    X_reg.insert(0, 'constant', 1) # insert constant
    X_t = X_reg.iloc[-1]
    X_reg = X_reg.values

    beta_OLS = inv(X_reg.T @ X_reg) @ X_reg.T @ Y_reg #OLS formula
    forecast = X_t @ beta_OLS
    return [beta_OLS, forecast]

#Estimating betas for INDPRO and forecasting:
print('Betas from regression of y = INDPRO and x = CPIAUCSL, TB3MS:', calculate_betas()[0])
print('Forecast from regression of y = INDPRO and x = CPIAUCSL, TB3MS:', calculate_betas()[1]*100)

# Estimating betas for CPIAUCSL and forecasting:
cpi_reg =  calculate_betas(target='CPIAUCSL', CoVars=['INDPRO', 'TB3MS'])
print('Betas from regression of y = CPIAUCSL and x = INDPRO, TB3MS:', cpi_reg[0])
print('Forecast from regression of y = CPIAUCSL and x = INDPRO, TB3MS:', cpi_reg[1]*100)

# Estimating betas for TB3ms and forecasting:
tb_reg =  calculate_betas(target='TB3MS', CoVars=['INDPRO', 'CPIAUCSL'])
print('Betas from regression of y = TB3MS and x = INDPRO, CPIAUCSL:', tb_reg[0])
print('Forecast from regression of y = TB3MS and x = INDPRO, CPIAUCSL:', tb_reg[1]*100)

## Estimation evaluation:
end_date = '12/1/1999'
H = [1,4,8]
p = 4
def MSFE(H, p, target, CoVars, end_date):
    ## Get the actual values of target at different steps ahead
    Y_actual = []
    Y_hat = []
    for h in H:
        os = pd.Timestamp(end_date) + pd.DateOffset(months=h)
        Y_actual.append(df_cleaned[df_cleaned['sasdate'] == os][target]*100)
        Y_hat = calculate_betas(h, p, target, CoVars, end_date)[1]*100
    return np.array(Y_actual) - np.array(Y_hat)


print(MSFE(H, 4, 'INDPRO', ['CPIAUCSL', 'TB3MS'], end_date))

t0 = pd.Timestamp('12/1/1999')
e = []
T = []
for j in range(0, 10):
    t0 = t0 + pd.DateOffset(months=1)
    print(f'Using data up to {t0}')
    ehat = MSFE(p = 4, H = [1,4,8], end_date = t0, target='INDPRO', CoVars=['CPIAUCSL', 'TB3MS'])
    e.append(ehat.flatten())
    T.append(t0)

## Create a pandas DataFrame from the list
edf = pd.DataFrame(e)
## Calculate the RMSFE, that is, the square root of the MSFE
print(np.sqrt(edf.apply(np.square).mean()))

from plotnine import *


def plotFE(H, p, target, CoVars, t0, range_end):
    t0 = pd.Timestamp(t0)
    e = []
    T = []
    for j in range(0, range_end):
        t0 = t0 + pd.DateOffset(months=1)
        ehat = MSFE(p=p, H=H,target=target, CoVars=CoVars, end_date=t0,)
        e.append(ehat.flatten())
        T.append(t0)

    edf = pd.DataFrame(e)
    print('The RMSE are', np.sqrt(edf.apply(np.square).mean()))

    plotlist = []
    for i in range(0, len(H)):
        plotdata = {'T' : T, 'Error' : edf.iloc[:,i]}
        plotdata = pd.DataFrame(plotdata)
        plotMSE = ggplot(plotdata) + aes(x = 'T', y = 'Error') + geom_line() + geom_point(size = 0.01) + theme_bw()
        plotlist.append(plotMSE)
    return plotlist

plotFE(H, p, 'INDPRO', ['CPIAUCSL', 'TB3MS'], t0 = '12/1/1999', range_end=10)[0].show()
