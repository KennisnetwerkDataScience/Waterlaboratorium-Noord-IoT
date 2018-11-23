# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 11:32:35 2018
@author: pahuizinga
"""
import pandas as pd

def BasicStatistics(dataframe):
    statframe = []
    statlist = []
    cols = dataframe.columns.tolist()
    for i in range(len(dataframe.columns)):
        if dataframe.iloc[:,i].dtypes == 'float' or dataframe.iloc[:,i].dtypes == 'int':
            statlist.append([cols[i], 'numeric', \
                             dataframe.iloc[:,i].min(), \
                             dataframe.iloc[:,i].max(), \
                             dataframe.iloc[:,i].max() - dataframe.iloc[:,i].min(), \
                             dataframe.iloc[:,i].mean(), \
                             dataframe.iloc[:,i].std(), \
                             dataframe.iloc[:,i].count(), \
                             dataframe.iloc[:,i].sum(), \
                             dataframe.iloc[:,i].nunique(), \
                             '',
                             dataframe.iloc[:,i].isnull().sum()])
        elif dataframe.iloc[:,i].dtypes == 'object':
            statlist.append([cols[i],'string', \
                             '' ,'' ,'' ,'' ,'' , \
                             dataframe.iloc[:,i].count(),'' , \
                             dataframe.iloc[:,i].nunique(), \
                             list(set(dataframe.iloc[:,i]))[:15], \
                             dataframe.iloc[:,i].isnull().sum()])
        elif dataframe.iloc[:,i].dtypes == 'datetime64[ns]':
            statlist.append([cols[i],'date', \
                             dataframe.iloc[:,i].min(), \
                             dataframe.iloc[:,i].max(), \
                             dataframe.iloc[:,i].max() - dataframe.iloc[:,i].min(), \
                             '', '' , \
                             dataframe.iloc[:,i].count(),'' , \
                             dataframe.iloc[:,i].nunique(), \
                             list(set(dataframe.iloc[:,i]))[:15], \
                             dataframe.iloc[:,i].isnull().sum()])    
        elif dataframe.iloc[:,i].dtypes == 'bool':
            statlist.append([cols[i],'boolean', \
                             '' ,'' ,'' ,'' ,'' , \
                             dataframe.iloc[:,i].count(), \
                             '' , \
                             dataframe.iloc[:,i].nunique(), \
                             '',
                             dataframe.iloc[:,i].isnull().sum()])
    columns = (['column', \
                'type', \
                'min', \
                'max', \
                'range', \
                'mean', \
                'stdev', \
                'aantal', \
                'sum', \
                'unique', \
                'uniquelist[max 15]', \
                'nan'])

    # Make dataframe
    statframe = pd.DataFrame(statlist, columns=columns)
    return statframe


def summarize_statistics(dictionairy):
    '''
    make summary of column statistics of an excel file with multiple sheets
    
        Parameters
        dictionairy: a dictionairy of excel sheets
    '''
    df_total = pd.DataFrame()
    for key in dictionairy.keys():

        df_new = BasicStatistics(dictionairy[key])
        # add excel sheetname to dataframe
        df_new['sheet'] = key
        
        df_total = df_total.append(df_new, ignore_index = True)
        
    return df_total
