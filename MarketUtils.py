#------ Market Data Functions ------#
import pandas as pd
from Configuration import *

#------ Reads data from csv file with ------#
def read_raw_data(filename, **kwargs):
    raw_data = pd.read_csv(filename, index_col = None).dropna()
    raw_data[DATE_COL] = pd.to_datetime(raw_data[DATE_COL], format=DATE_FORMAT)
    return raw_data

#------ Get data for a few dates -------#
def get_data_for_interval(filename,termDates,**kwargs):
    raw_data = read_raw_data(filename, **kwargs)
    mask = (raw_data[DATE_COL] > termDates[0] and raw_data[DATE_COL] < termDates[1])
    return raw_data.loc[mask]


#------ Data cleaning tasks --------#
def clean_raw_data(data): pass



