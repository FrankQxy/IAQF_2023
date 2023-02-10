import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def arrow_plot(price_data, trading_signals):
    """
    data price plot marked with long/short/inactive positions
    ---------------------------
    MARKER COLOR
    red -> long asset1, short asset2
    blue -> short asset1, long asset2
    green -> inactive
    ----------------------------
    :param price_data: (format) standard dataframe of indices price columns
    :param positions:  (format) standard dataframe of indices trading signal columns
    """
    colors = {1:'red', 0:'green', -1:'blue'}
    idx0, idx1 = price_data.columns

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(price_data, color='black')
    ax.scatter(price_data.index, price_data.iloc[:, 0],
               c=np.sign(trading_signals.iloc[:, 0]).map(colors), zorder=3)
    ax.scatter(price_data.index, price_data.iloc[:, 1],
               c=np.sign(trading_signals.iloc[:, 0]).map(colors), zorder=3)
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
    plt.legend([idx0, idx1])
    plt.title('Position Analysis Plot')
    ax.set_xlabel(f'Red maker: long {idx0}, short {idx1}\n Blue maker: short {idx0}, long{idx1}\n Green marker: inactive')
    
