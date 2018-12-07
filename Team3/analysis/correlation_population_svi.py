import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import copy
import argparse

from .util import *
from .load import *


def reindex_interpolate(df, index):
    df = df.reindex(index = index)
    df = df.interpolate(method = 'linear', inplace = True)
    return df


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


def save_plot_population_svi(data, file_name, legend=None, svi='SVI 30'):
    plt.rcParams["figure.figsize"] = (20,15)
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    data['population'].plot(ax=ax1, logy=False, legend=legend, style='.-', title=file_name)
    data['svi'][svi].plot(ax=ax2, legend=False, color='black', linewidth=2, style='.-')
    fig.savefig(os.path.join('analysis_result', file_name), dpi=fig.dpi)


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--start', type=str, help='start', default='2014-10-1')
parser.add_argument('--end', type=str, help='end', default='2017-3-1')
parser.add_argument('--svi', type=str, help='svi column', default='SVI 30')
parser.add_argument('--min', type=float, help='min population', default=.000001)
parser.add_argument('--mincorr', type=float, help='min correlation', default=.6)
parser.add_argument('--maxcorr', type=float, help='max correlation', default=1)
parser.add_argument('--method', type=str, help='correlation method', default='pearson')#, values=['pearson', 'kendall', 'spearman'])
parser.add_argument('--corron', type=str, help='correlation on', default='')#, values=['', 'log10'])

args = parser.parse_args()

start = args.start#'2014-10-1'
end = args.end#'2017-3-1'
population_minimum = args.min#0.000001
svi = args.svi#'SVI 5'
l_min = args.mincorr#.57
l_max = args.maxcorr#1
corron=args.corron
method=args.method

data = load(['population', 'svi'])
data['population'] = data['population'].fillna(population_minimum)

d = data['population']
p_min = data['population'].min()
p_max = data['population'].max()
p_frac = p_max / p_min
#print(d)
#print(p_frac)
#print(p_frac > 100)
sel = p_frac > 100
data['population'] = data['population'][sel.index[sel]]
#print (data['population'])

if corron == 'log10':
    data['population'] = data['population'].apply(np.log10)
if corron == 'log':
    data['population'] = data['population'].apply(np.log)
df = population_svi_one_df_interpolate(data, svi)
print(df)
s_corr = df.corrwith(df[svi])
s_corr.sort_values(inplace=True)
#print(s_corr.dropna())

sel = s_corr.dropna().drop(data['svi'].columns)[(s_corr > l_min) & (s_corr < l_max)].index
#print(sel)
data_c = copy.copy(data)
data_c['population'] = data['population'][sel]
data_c = limit_range(data_c, start, end)
data_c['svi'] = df
#save_plot_population_svi(limit_range(data_c, start, end), 'population_svi_%s_%s_%s_%s_%s_>%s_<%s.png' % (start, end, svi, population_minimum, corron, l_min, l_max), legend='top', svi=svi)
save_plot_population_svi(data_c, 'population_svi_%s_%s_%s_%s_%s_>%s_<%s.png' % (start, end, svi, population_minimum, corron, l_min, l_max), legend='top', svi=svi)

df = pd.concat([data_c['population'], data_c['svi']], axis=1, sort=True)
print(df)
matrix = df.corr(method='pearson')
print(matrix['SVI 30'].sort_values())
print(len(data_c['population'].columns))
print(len(data_c['svi'].columns))
'''
l_min = -1
l_max = -.4
sel = s_corr.dropna().drop(data['svi'].columns)[(s_corr > l_min) & (s_corr < l_max)].index
print(sel)
data_c = copy.copy(data)
data_c['population'] = data['population'][sel]
save_plot_population_svi(limit_range(data_c, start, end), 'population_svi_%s_%s_%s_%s_>%s_<%s.png' % (start, end, svi, population_minimum, l_min, l_max), legend='top', svi=svi)
'''

'''
data = load(['population', 'svi'])
data['population'] = data['population'].fillna(0.00000001)
data['population'] = data['population'].groupby(axis=1, level=[0,1,2,3]).sum()
print(data['population'])

df = population_svi_one_df_interpolate(data, 'SVI 30')
s_corr = df.corrwith(df['SVI 30'])
s_corr.sort_values(inplace=True)
print(s_corr.dropna())

l_min = .65
l_max = 1
sel = s_corr.dropna().drop(data['svi'].columns)[(s_corr > l_min) & (s_corr < l_max)].index
print(sel)
data_c = copy.copy(data)
data_c['population'] = data['population'][sel]
save_plot_population_svi(limit_range(data_c, start, end), 'population_svi_%s_%s_%s_>%s_<%s.png' % (start, end, 'l2', l_min, l_max), legend='top')

l_min = -1
l_max = -.4
sel = s_corr.dropna().drop(data['svi'].columns)[(s_corr > l_min) & (s_corr < l_max)].index
print(sel)
data_c = copy.copy(data)
data_c['population'] = data['population'][sel]
save_plot_population_svi(limit_range(data_c, start, end), 'population_svi_%s_%s_%s_%s.png' % (start, end, 'l2', '<-0.40'), legend=True)
'''
