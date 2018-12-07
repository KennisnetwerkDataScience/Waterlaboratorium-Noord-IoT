import pandas as pd
import matplotlib.pyplot as plt
import math
import copy

from .util import *
from .load import *
from .interpolate import *


def plot(data, **kwargs):
    plt.rcParams["figure.figsize"] = (20,15)
    fig, axo = plt.subplots()
    legend=True

    for i, key in enumerate(data):
        ax = axo.twinx()
        print(type(data[key].index))
        print(data[key].index)
        kwargs_ = kwargs[key]
        data[key].plot(ax=ax, **kwargs_)
    plt.savefig(os.path.join('analysis_result', 'knmi_daily_mean_svi.png'), dpi=fig.dpi)
    #plt.show()


start = '2014-10-1'
end = '2017-3-1'

data = load(['knmi_daily_mean', 'svi'])
data = limit_range(data, start, end)

data['knmi_daily_mean'] = data['knmi_daily_mean'][['T']]
data['svi'] = data['svi'][['SVI 30', 'SVI 5']]

params = {
    'knmi_daily_mean': {
        'logy': False,
        'legend': True,
        'style': '.-',
        'color': 'black',
    },
    'svi': {
        'logy': True,
        'legend': True,
        'style': '.-',
    }
}
plot(data, **params)
