import pandas as pd
import matplotlib.pyplot as plt
import math

from .util import *
from .load import *


data = load(['population', 'svi'])
#data['population'] = data['population'].fillna(0.00000001)
data['population'] = data['population'].groupby(axis=1, level=[0,1,2,3]).sum()

start = '2014-10-1'
end = '2017-2-1'
save_plot_population_svi(limit_range(data, start, end), 'population_svi_%s_%s.png' % (start, end))

start = '2016-4-1'
end = '2016-8-1'
save_plot_population_svi(limit_range(data, start, end), 'population_svi_%s_%s.png' % (start, end))
