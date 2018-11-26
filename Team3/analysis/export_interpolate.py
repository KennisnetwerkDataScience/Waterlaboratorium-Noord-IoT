import pandas as pd
import matplotlib.pyplot as plt
import math
import copy

from .util import *
from .load import *


start = '2014-10-1'
end = '2017-3-1'
min_population = 0.000001

def export(data):
    for key in data:
        data[key].to_csv('export/%s.csv' % key)


data = load(['svi', 'influent', 'effluent', 'population', 'knmi'])
#data['population'] = data['population'].fillna(min_population)
data['knmi_mean'] = data['knmi'].groupby(pd.Grouper(freq='D')).mean().dropna()
data['knmi_sum'] = data['knmi'].groupby(pd.Grouper(freq='D')).sum().dropna()

data = remove_index_duplicates(data)
data = limit_range(data, start, end)

data['svi'] = reindex_interpolate(data['svi'], data['population'])
data['influent'] = reindex_interpolate(data['influent'], data['population'])
data['effluent'] = reindex_interpolate(data['effluent'], data['population'])
data['knmi_sum'] = reindex_interpolate(data['knmi_sum'], data['population'])
data['knmi_mean'] = reindex_interpolate(data['knmi_mean'], data['population'])

export(data)
