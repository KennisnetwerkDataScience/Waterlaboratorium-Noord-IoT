import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import copy
import argparse
import os, errno

from .util import *
from .load import *
from .interpolate import *


def ensure_dir(dir):
    try:
        os.makedirs(dir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

def save_plot_population_svi(data, file_name, legend=None, svi='SVI 30'):
    plt.rcParams["figure.figsize"] = (20,15)
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    data['population'].plot(ax=ax1, logy=False, legend=legend, style='.-', title=file_name)
    data['svi'][svi].plot(ax=ax2, legend=False, color='black', linewidth=2, style='.-')
    fig.savefig(os.path.join('analysis_result', file_name), dpi=fig.dpi)

def get(kwargs, arg, l):
    return kwargs[arg] if arg in kwargs else [None] * l

def save_plot(dfs, dir, name, **kwargs):
    plt.rcParams["figure.figsize"] = (20,15)
    fig, ax1 = plt.subplots()
    title = dir + name
    legends = get(kwargs, 'legends', len(dfs))
    styles = get(kwargs, 'styles', len(dfs))
    colors = get(kwargs, 'colors', len(dfs))
    linewidths = get(kwargs, 'linewidths', len(dfs))
    for idx,df,legend,style,color, linewidth in zip(range(len(dfs)),dfs, legends, styles, colors, linewidths):
        ax = ax1.twinx() if idx == 0 else ax1
        kwargs = {}
        if legend:
            kwargs['legend'] = legend
        if style:
            kwargs['style'] = style
        if color:
            kwargs['color'] = color
        if linewidth:
            kwargs['linewidth'] = linewidth
        df.plot(ax=ax, title=title, **kwargs)
        #data['svi'][svi].plot(ax=ax2, legend=False, color='black', linewidth=2, style='.-')
    ensure_dir(os.path.join('analysis_result', dir))
    fig.savefig(os.path.join('analysis_result', dir, '{}.png'.format(name)), dpi=fig.dpi)
    plt.close()



def fill(df, v):
    return df.fillna(population_minimum)

def select_population_factor(df, v):
    p_min = df_p.min()
    p_max = df_p.max()
    p_frac = p_max / p_min
    sel = p_frac > 100
    return df_p[sel.index[sel]]

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--start', type=str, help='start', default='2014-10-1')
parser.add_argument('--end', type=str, help='end', default='2017-3-1')
parser.add_argument('--min', type=float, help='min population', default=.000001)
parser.add_argument('--method', type=str, help='correlation method', default='pearson', choices=['pearson', 'kendall', 'spearman'])
parser.add_argument('--scale', type=str, help='scale population', default='none', choices=['none', 'log', 'log10'])

args = parser.parse_args()

start = args.start#'2014-10-1'
end = args.end#'2017-3-1'
population_minimum = args.min#0.000001
scale=args.scale
method=args.method

data = load(['population', 'svi'])
df_p = data['population']
df_s = data['svi']

df_p = df_p.fillna(population_minimum)
df_p = select_population_factor(df_p, 100)

#if scale != 'none':
#    f = getattr(np, scale)
#    df_p = df_p.apply(f)

if scale == 'log10':
    df_p = df_p.apply(np.log10)
if scale == 'log':
    df_p = df_p.apply(np.log)

df_s = reindex_interpolate(df_s, df_p)
df = pd.concat([df_p, df_s], axis=1, sort=True)
matrix = df.corr(method=method)

for pc in df_p.columns:
    df_p_c = copy.copy(df_p[[pc]])
    for sc in df_s.columns:
        df_s_c = copy.copy(df_s[[sc]])
        print(pc)
        print(sc)
        print(df_p_c)
        corr = matrix.loc[[pc],sc]
        save_plot(
            [df_p_c, df_s_c],
            '%s_%s_%s_%s_%s' % (start, end, population_minimum, scale, method),
            '%+1.5f_%s_%s' % (corr, pc[:-1], sc),
            legends=['top left', 'top right'],
            colors=['red', 'black'],
            linewidths=[2,2],
            styles=['-', '.-'],
        )
