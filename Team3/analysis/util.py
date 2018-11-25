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
