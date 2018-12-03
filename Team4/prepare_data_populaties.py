# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 11:14:28 2018

@author: Pieter

Prepares the data in the Kennisnetwerk_data_WWTPpopulaties_WLN.xlsx file

"""

import pandas as pd
from pathlib import Path

file_location = Path(r'E:\Files\wln')
output_file_location  = Path(r'E:\Files\wln\output')


df_pop_arc = pd.read_excel(file_location / 'Kennisnetwerk_data_WWTPpopulaties_WLN.xlsx', sheet_name='Pop_BAC_ARC_EUK_meetup')

# datum is 1st day of the week
df_pop_arc['datum'] = pd.to_datetime(df_pop_arc['Weeknumber'].astype(str)+  df_pop_arc['Year_Sample'].astype(str).add('-0') ,format='%W%Y-%w')
df_pop_arc['genera_join'] = df_pop_arc['Parameter'].str.lower()


df_bact_family = pd.read_excel(file_location / 'Kennisnetwerk_data_WWTPpopulaties_WLN.xlsx', sheet_name='Pivot_ZAWZI_Ranked', skiprows=9, usecols='A:D')
df_bact_family['genera_join'] = df_bact_family['Genera'].str.lower()

df_samples = pd.merge(df_pop_arc, df_bact_family, how='left', on='genera_join')

# not all can be joined?
df_samples = df_samples.drop(columns=['genera_join'])

df_samples.to_excel(output_file_location / 'bact_samples.xlsx')