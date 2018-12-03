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


def save_plot_population_svi(data, file_name, legend=None):
    plt.rcParams["figure.figsize"] = (20,15)
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    data['population'].plot(ax=ax1, logy=True, legend=legend, style='.-', title=file_name)
    data['svi']['SVI 30'].plot(ax=ax2, legend=legend, color='black', linewidth=2, style='.-')
    fig.savefig(os.path.join('analysis_result', file_name), dpi=fig.dpi)


def df_remove_index_duplicates(df, keep='last'):
    return df[~df.index.duplicated(keep=keep)]


def remove_index_duplicates(data, keep='last'):
    return {k: df_remove_index_duplicates(data[k], keep=keep) for k in data}


def reindex_interpolate(df, df_target):
    l = df_target.index.tolist()
    #print(l)
    l.extend(df.index.tolist())
    l = sorted(list(set(l)))
    #print(l)
    df = df.reindex(index = l)
    #print(df)
    df = df.interpolate()
    df = df.reindex(index = df_target.index)
    return df


def correlations(data, target_table, target_column):
    return {key: data[key].corrwith(data[target_table][target_column]) for key in data}
