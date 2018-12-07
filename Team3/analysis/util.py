import pandas as pd
import matplotlib.pyplot as plt
import math
import os.path

def limit_df_index(df, start, end):
    return df.loc[(df.index > start) & (df.index <= end)]


def limit_range(data, start, end):
    return {key: limit_df_index(data[key], start, end) for key in data}


def check_duplicate_columns(df):
    cl = list(df.columns)
    cs = set(cl)
    for i1 in cs:
        for idx, i2 in enumerate(cl):
            if i1 == i2:
                del cl[idx]
                break
    if cl:
        raise Exception('Duplicate columns %s' % cl)


def save_plot_population_svi(data, file_name, legend=None, svi='SVI 30'):
    plt.rcParams["figure.figsize"] = (20,15)
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    data['population'].plot(ax=ax1, logy=True, legend=legend, style='.-', title=file_name)
    data['svi'][svi].plot(ax=ax2, legend=legend, color='black', linewidth=2, style='.-')
    fig.savefig(os.path.join('analysis_result', file_name), dpi=fig.dpi)


def save_check_plot(dfs, dir_name, file_name, legend=None, styles=None, **kwargs):
    plt.rcParams["figure.figsize"] = (20,15)
    fig, ax1 = plt.subplots()
    axes = [ax1]
    for i,df in enumerate(dfs):
        ax = ax1 if i == 0 else ax1.twinx()
        style = styles[i] if styles else '.-'
        df.plot(ax=ax, legend=legend, style=style, title=file_name, **kwargs)
    fig.savefig(os.path.join(dir_name, file_name), dpi=fig.dpi)


def df_remove_index_duplicates(df, keep='last'):
    return df[~df.index.duplicated(keep=keep)]


def remove_index_duplicates(data, keep='last'):
    return {k: df_remove_index_duplicates(data[k], keep=keep) for k in data}


def df_index_duplicates(df):
    return df.index.duplicated()


def index_duplicates(data):
    return {k: df_index_duplicates(data[k]) for k in data}


def correlations(data, target_table, target_column):
    return {key: data[key].corrwith(data[target_table][target_column]) for key in data}


def flatten_columns(df):
    df.columns = [''.join(str(col)).strip() for col in df.columns.values]
