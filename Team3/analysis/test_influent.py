import pandas as pd
import matplotlib.pyplot as plt
import math
import copy

from .util import *
from .load import *


start = '2014-10-1'
end = '2017-3-1'
min_population = 0.000001

plt.rcParams["figure.figsize"] = (20,15)
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
legend=True

data = load(['influent', 'svi'])
#data['population'] = data['population'].fillna(min_population)
#data['knmi'].groupby(pd.Grouper(freq='D')).mean().dropna()
data = remove_index_duplicates(data)

data = limit_range(data, start, end)

print(data['svi'])
data['svi']['SVI 30'].plot(ax=ax1, legend=legend, color='black', linewidth=2, style='.-')

#data['influent'] = reindex_interpolate(data['influent'], data['svi'])
print(data['influent'])
data['influent'].plot(ax=ax2, logy=True, legend=legend, linewidth=1, style='-')


plt.show()
#fig.savefig(os.path.join('analysis_result', file_name), dpi=fig.dpi)
