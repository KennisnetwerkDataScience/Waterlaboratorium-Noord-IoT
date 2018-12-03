# -*- coding: utf-8 -*-
"""
Created on Sat Nov 24 16:56:30 2018

@author: Pieter

Prepares the data in the Kennisnetwerk_data_WWTPpopulaties_WLN.xlsx file sheet Pivot_ZAWZI_Ranked

"""

import pandas as pd
from pathlib import Path

file_location = Path(r'E:\Files\wln')
output_file_location  = Path(r'E:\Files\wln\output')


df_bact_family = pd.read_excel(file_location / 'Kennisnetwerk_data_WWTPpopulaties_WLN.xlsx', sheet_name='Pivot_ZAWZI_Ranked', skiprows=9, usecols='A:AP')

# the information is in a pivot table, the year/week information is partly in a merged cell
# So use a second dataframe to get the year and weeks
df_column_info = pd.read_excel(file_location / 'Kennisnetwerk_data_WWTPpopulaties_WLN.xlsx', sheet_name='Pivot_ZAWZI_Ranked', skiprows=4, usecols='K:AP')

def get_weeks(df):
    result = list()
    
    for column in df.columns:
        # the year is in a merged cell, so remember this as the current year
        if type(column) is int:
            cur_year = column
    
        week = df_column_info.loc[1, column]
        
        result.append("{:0}{:02}".format(cur_year, week))    
    return result


week_columns = get_weeks(df_column_info)

all_columns = list(df_bact_family.columns)


assert len(all_columns[10:]) == len(week_columns), "column length should be equal"


updated_columns = all_columns[0:10] + week_columns

df_bact_family.columns = updated_columns

df_bact_family.to_excel(output_file_location / 'bact_family.xlsx')
