def arrow_plot(price_data, positions):
    """
    data price plot marked with long/short/inactive positions
    ---------------------------
    MARKER COLOR
    red -> long asset1, short asset2
    blue -> short asset1, long asset2
    green -> inactive
    ----------------------------
    :param price_data: (format) dataframe of two columns
    :param positions:  (format) dataframe of two columns
    :return:
    """
    colors = {1:'red', 0:'green', -1:'blue'}

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(price_data, color='black')
    ax.scatter(price_data.index, price_data.iloc[:, 0],
               c=np.sign(positions.iloc[:, 0]).map(colors), zorder=3)
    ax.scatter(price_data.index, price_data.iloc[:, 1],
               c=np.sign(positions.iloc[:, 0]).map(colors), zorder=3)
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
