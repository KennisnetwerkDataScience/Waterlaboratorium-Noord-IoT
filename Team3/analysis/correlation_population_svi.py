import pandas as pd
import matplotlib.pyplot as plt
import math
import copy

from .util import *
from .load import *


def population_svi_one_df_interpolate(data, svi_column):
    df = data['population']
    data['population'].columns = [''.join(str(col)).strip() for col in data['population'].columns.values]
    check_duplicate_columns(data['population'])
    df = df.combine_first(data['svi'])
    df = df[~df.index.duplicated(keep='last')]
    df.sort_index(inplace=True)
    df = df.interpolate('index')
    df = df.reindex(data['population'].index)
    return df


start = '2014-10-1'
end = '2017-3-1'

data = load(['population', 'svi'])
data['population'] = data['population'].fillna(0.00000001)

df = population_svi_one_df_interpolate(data, 'SVI 30')
s_corr = df.corrwith(df['SVI 30'])
s_corr.sort_values(inplace=True)
print(s_corr.dropna())

sel = s_corr.dropna().drop(data['svi'].columns)[s_corr > .75].index
print(sel)
data_c = copy.copy(data)
data_c['population'] = data['population'][sel]
save_plot_population_svi(limit_range(data_c, start, end), 'population_svi_%s_%s_%s.png' % (start, end, '>0.75'), legend='top')

sel = s_corr.dropna().drop(data['svi'].columns)[s_corr < -.40].index
print(sel)
data_c = copy.copy(data)
data_c['population'] = data['population'][sel]
save_plot_population_svi(limit_range(data_c, start, end), 'population_svi_%s_%s_%s.png' % (start, end, '<-0.40'), legend=True)



data = load(['population', 'svi'])
data['population'] = data['population'].fillna(0.00000001)
data['population'] = data['population'].groupby(axis=1, level=[0,1,2,3]).sum()
print(data['population'])

df = population_svi_one_df_interpolate(data, 'SVI 30')
s_corr = df.corrwith(df['SVI 30'])
s_corr.sort_values(inplace=True)
print(s_corr.dropna())

sel = s_corr.dropna().drop(data['svi'].columns)[s_corr > .75].index
print(sel)
data_c = copy.copy(data)
data_c['population'] = data['population'][sel]
save_plot_population_svi(limit_range(data_c, start, end), 'population_svi_%s_%s_%s_%s.png' % (start, end, 'l2', '>0.75'), legend='top')

sel = s_corr.dropna().drop(data['svi'].columns)[s_corr < -.40].index
print(sel)
data_c = copy.copy(data)
data_c['population'] = data['population'][sel]
save_plot_population_svi(limit_range(data_c, start, end), 'population_svi_%s_%s_%s_%s.png' % (start, end, 'l2', '<-0.40'), legend=True)
