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
legend=False

data = load(['population', 'svi'])
data['population'] = data['population'].fillna(min_population)
data = remove_index_duplicates(data)


print(data['svi'])
data['svi']['SVI 30'].plot(ax=ax1, legend=legend, color='black', linewidth=2, style='.-')

data['svi'] = reindex_interpolate(data['svi'], data['population'])
print(data['svi'])
data['svi']['SVI 30'].plot(ax=ax1, legend=legend, color='red', linewidth=2, style='.-')


plt.show()
#fig.savefig(os.path.join('analysis_result', file_name), dpi=fig.dpi)
