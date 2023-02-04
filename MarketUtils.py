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
def read_raw_data(filename, **kwargs):
    raw_data = pd.read_csv(filename, index_col = None).dropna()
    raw_data[DATE_COL] = pd.to_datetime(raw_data[DATE_COL], format=DATE_FORMAT)
    return raw_data

#----- read index data -------#
def read_index_data(**kwargs):
    raw_prices = read_raw_data(INDEX_DATA, **kwargs)
    return raw_prices

#------ Get data -------#
def get_data(**kwargs):

    col_list = ["Date"]

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