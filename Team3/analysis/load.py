import pandas as pd
import matplotlib.pyplot as plt
import math
from datetime import datetime

def ilog(x):
    return 10 ** x


def num(x):
    print('`%s` %s' % (x, type(x)))
    if type(x) == str:
        if not x.strip():
            return float('NaN')
        x = float(''.join(c for c in x if c.isdigit() or c == '.'))

    if math.isnan(x):
        return x
    return int(x)


def load_population():
    df = pd.read_excel(
        '../Data/Kennisnetwerk_data_WWTPpopulaties_WLN_no_password.xlsx',
        skiprows=[0,1,2,3,4,5,6,7,8,9],
        header=[0,1],
        usecols='A:E,K:AP',
        index_col=[0,1,2,3,4],
    )
    df = df.transpose()
    df.index = pd.Index(pd.datetime(year, 1, 1) + pd.Timedelta(7 * weeks, unit='d') for year, weeks in df.index)
    df = df.applymap(ilog)
    return df


def load_effluent():
    df = pd.read_excel(
        '../Data/Kennisnetwerk_procesdata_kwaliteitsdata_WLN_no_password.xlsx',
        sheet_name='effl. lab analyses',
        header=[0],
        index_col=[0],
    )
    df.index = pd.to_datetime(df.index)
    df = df.applymap(num)
    return df


def load_influent():
    df = pd.read_excel(
        '../Data/Kennisnetwerk_procesdata_kwaliteitsdata_WLN_no_password.xlsx',
        sheet_name='Influent lab analyses',
        header=[0],
        index_col=[0],
    )
    df.index = pd.to_datetime(df.index)
    df = df.applymap(num)
    return df


def load_svi():
    df = pd.read_excel(
        '../Data/Kennisnetwerk_procesdata_kwaliteitsdata_WLN_no_password.xlsx',
        sheet_name='SVI en DS AT',
        header=[0],
        index_col=[0],
    )
    df.index = pd.to_datetime(df.index)
    return df


def load_online():
    df = pd.read_excel(
        '../Data/Kennisnetwerk_procesdata_kwaliteitsdata_WLN_no_password.xlsx',
        sheet_name='online data',
        header=[0],
        index_col=[0],
    )
    df.index = pd.to_datetime(df.index)
    return df


def load_knmi():
    skiprows = list(range(30))
    skiprows.append(32)
    df = pd.read_csv(
        '../Data/uurgeg_286_2011-2020.txt',
        skiprows=skiprows,
        header=0,
        #index_col=1,
    )
    print(df)
    print(df.columns)
    df.columns = [c.strip() for c in df.columns]
    print(df.columns)
    df.index = df.apply(
        lambda e : pd.datetime.combine(
            datetime.strptime(str(e['YYYYMMDD']), '%Y%m%d').date(),
            datetime.strptime(str(e['HH']-1), '%H').time()
        ),
        axis=1
    )
    return df


def load(tables):
    return {table: globals()['load_' + table]() for table in tables}


def load_process():
    df_svi = load_svi()
    df_effl = load_effluent()
    df = df_svi.append(df_effl, sort=True)
    df.sort_index(inplace=True)
    return df
