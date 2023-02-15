import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm


def clean_data():
    spread = pd.read_csv('./Benchmark/BenchmarkSpread.csv')
    vix_data = pd.read_csv('./DataFiles/vix_data.csv')
    yld = pd.read_csv('./DataFiles/yield.csv')

    vix_data.drop_duplicates(inplace=True)
    yld.drop_duplicates(inplace=True)

    dates = spread['Date']
    vix_data.set_index(['Date'], inplace=True)
    vix_data = vix_data.loc[dates].fillna(method='ffill')

    yld['Date'] = yld['Date'].astype(str)
    yld['Date'] = yld['Date'].str[0:4] + '-' + yld['Date'].str[4:6] + '-' + yld['Date'].str[6:]
    yld.set_index(['Date'], inplace=True)

    fedfunds = pd.DataFrame(index=dates)
    arr = np.zeros(len(dates))
    for i in range(len(dates)):
        if dates[i] in yld.index:
            arr[i] = yld['fedfunds'][dates[i]]
    fedfunds['Fedfunds'] = arr
    fedfunds = fedfunds.fillna(method='ffill')

    df = pd.DataFrame()
    df['Dates'] = dates.values
    df['Vix'] = vix_data['^GSPCVIX'].values
    df['Fedfunds'] = fedfunds['Fedfunds'].values
    df['Spread'] = spread['Spread'].values

    return df


def linear_regression():
    df = clean_data()
    X = df[["Vix", "Fedfunds"]]
    y = df["Spread"]
    X = sm.add_constant(X)

    model = sm.OLS(y, X).fit()
    return model.summary()


