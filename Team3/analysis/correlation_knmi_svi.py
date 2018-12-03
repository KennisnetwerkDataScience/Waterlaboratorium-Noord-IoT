import pandas as pd
import matplotlib.pyplot as plt
import math
import copy

from .util import *
from .load import *


def knmi_svi_one_df_interpolate(data, svi_column):
    df = data['knmi']
    df = df.combine_first(data['svi'])
    df = df[~df.index.duplicated(keep='last')]
    df.sort_index(inplace=True)
    df = df.interpolate('index')
    df = df.reindex(data['knmi'].index)
    return df


start = '2014-10-1'
end = '2017-2-1'

data = load(['knmi', 'svi'])
print(data['knmi'])
df = knmi_svi_one_df_interpolate(data, 'SVI 30')
s_corr = df.corrwith(df['SVI 30'])
s_corr.sort_values(inplace=True)
print(s_corr)
