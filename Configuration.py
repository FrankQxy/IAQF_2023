# Module to keep constants and hard-coded values
# as well as paths and security names

DATE_COL = 'Date'
DATE_FORMAT = r'%Y-%m-%d'

import os
from datetime import datetime
cwd = os.getcwd()

DATA_PATH = cwd + '\DataFiles'
INDEX_DATA = DATA_PATH + '\index_prices.csv'
BASE_DIR = cwd

TIMESTAMP = lambda : datetime.now().strftime("%Y%m%d")

INDEX_LIST = ['^GSPC', '^IXIC', '^DJI','^RUA']

FUNC_BIND = {'index' : 'read_index_data'}