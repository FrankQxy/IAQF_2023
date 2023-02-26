# Module to keep constants and hard-coded values
# as well as paths and security names

DATE_COL = 'Date'
DATE_FORMAT = r'%d-%m-%Y'

import os
from datetime import datetime
from pathlib import Path
cwd = Path(os.getcwd())

DATA_PATH = cwd / "DataFiles"
INDEX_DATA = DATA_PATH  / "index_prices.csv"
print(INDEX_DATA)
VIX_DATA = DATA_PATH / "vix_data.csv"
YIE_DATA = DATA_PATH / "yield.csv"
SENTI_DATA = DATA_PATH / "news_sentiment_data.csv"
BASE_DIR = cwd

TIMESTAMP = lambda : datetime.now().strftime("%Y%m%d")

INDEX_LIST = ['^GSPC', '^IXIC', '^DJI']
YIELD_LIST = ['fedfunds','Yield_1Y','Yield_2Y',	'Yield_5Y',	'Yield_7Y',	'Yield_10Y','Yield_20Y','Yield_30Y']
SENTI_LIST = ['News Sentiment']
VIX_LIST = ['^GSPC', '^IXIC', '^DJI']

FUNC_BIND = {'index' : 'read_index_data', 'vix' : 'read_vix_data','alter' : 'read_alter_data', 'news' : 'read_news_data', 'yield' : 'read_yield_data'}