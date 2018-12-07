import pandas as pd


method='linear'

def reindex_interpolate_old(df, df_target):
    l = df_target.index.tolist()
    #print(l)
    l.extend(df.index.tolist())
    l = sorted(list(set(l)))
    #print(l)
    df = df.reindex(index = l)
    #print(df)
    df = df.interpolate(method='time')
    df = df.reindex(index = df_target.index)
    return df


def reindex_interpolate(df, df_target, **kwargs):
    date_range = pd.date_range(start=df.index[0], end=df.index[-1], freq='D')
    df = df.reindex(date_range)
    #print(df)
    for col in df:
        try:
            df[col] = df[col].interpolate(**kwargs)
        except Exception as ex:
            print('Exception at column %s: %s' % (col,ex))
    df = df.reindex(index = df_target.index)
    #print(df)
    return df


def interpolate(data, table, **kwargs):
    return {k: reindex_interpolate(data[k], data[table], **kwargs) for k in data if k != table}
