#------ Market Data Functions ------#
import pandas as pd
from Configuration import *
import yfinance as yf
from datetime import datetime, timedelta

#------- download index data ------#
def download_index_data():
    prices = yf.download(INDEX_LIST).dropna()['Adj Close']
    prices.to_csv(INDEX_DATA)

#------ Reads data from csv file with ------#
def read_raw_data(filename, **kwargs) -> pd.DataFrame:
    raw_data = pd.read_csv(filename, index_col = DATE_COL).dropna()
    # raw_data.index = raw_data.index.strftime(DATE_FORMAT)
    return raw_data

#----- read index data -------#
def read_index_data(**kwargs) -> pd.DataFrame:
    raw_prices = read_raw_data(INDEX_DATA, **kwargs)
    return raw_prices

#------ Get data -------#
def get_data(**kwargs) -> pd.DataFrame:
    """Primary function to get data from csv
    Args:
    kwargs['type'] (str, optional): indicates the type of data needed like "index"
    kwargs[[start_date,end_date]] (list, optional): start and end date
    kwargs["col_list"]: list of columns need in the data other than date
    Will add more as the need arises

    Returns:
        pd.DataFrame: DataFrame object with index column date
    """
    col_list = []
    if 'type' in kwargs:
       type = kwargs['type']
    else :
       type = "index"

    if type == 'index':
       if 'col_list' in kwargs:
           col_list += kwargs['col_list']
       else:
           col_list += INDEX_LIST

    data_reader = globals()[FUNC_BIND[type]]
    raw_data = data_reader(**kwargs)[col_list]

    if 'termDates' in kwargs:
        termDates = kwargs['termDates']
        mask = (raw_data[DATE_COL] > termDates[0] and raw_data[DATE_COL] < termDates[1])
        return raw_data.loc[mask]
    
    return raw_data

#------ Data cleaning tasks --------#
def clean_raw_data(data): pass