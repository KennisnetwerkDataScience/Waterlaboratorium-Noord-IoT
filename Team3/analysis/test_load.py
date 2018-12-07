import pandas as pd
import matplotlib.pyplot as plt
import math
import copy

from .util import *
from .load import *


def print_any_na(data):
    for k in data:
        df = data[k]
        print(k)
        print(df[df.isnull().any(axis=1)])


def any_na(data):
    return {k: df[df.isnull().any(axis=1)] for k, df in data.items()}


tables = ['influent', 'effluent', 'svi', 'population']
tables = ['svi']

print('load')
data = load(tables)

print('===')
print('any_na')
print(any_na(data))

print('===')
print('index duplicates')
duplicates = index_duplicates(data)
gen = ({key:duplicates[key]} for key in duplicates if duplicates[key].any())
for x in gen:
    print(x)
