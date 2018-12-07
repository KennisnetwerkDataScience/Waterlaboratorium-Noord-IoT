import pandas as pd
import matplotlib.pyplot as plt
import math
import copy

from .util import *
from .load import *
from .interpolate import *

def get_method_args(method):
    if not method in method_args:
        return [{}]
    return method_args[method]


tables = ['influent', 'effluent', 'svi', 'population']
targets = tables
methods = ['linear', 'nearest', 'slinear', 'quadratic', 'cubic', 'barycentric', 'polynomial', 'spline']
method_args = {
    'spline':[{'order':2}],
    'polynomial':[{'order':2}]
}

'''
tables = ['influent', 'effluent']
targets = ['effluent']
methods = ['polynomial', 'spline','linear']
'''

print('load')
data = load(tables)
data = remove_index_duplicates(data)

for target in targets:
    for method in methods:
        for args in get_method_args(method):
            print('interpolate on %s with %s %s' % (target, method, args))
            kwargs = {'method':method}
            kwargs.update(args)
            interpolated = interpolate(data, target, **kwargs)
            for k in interpolated:
                save_check_plot([data[k], interpolated[k]], 'result_test_interpolate', '%s_on_%s_%s_%s.png' % (k, target, method, args), styles=['.', '-'], legend=True, logy=True)
