# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 14:05:41 2018

@author: Pieter


Uses the files created with prepare_data.py and prepare_data_polutaties.py

"""

import pandas as pd
from pathlib import Path
from sklearn import preprocessing

output_file_location  = Path(r'E:\Files\wln\output')


min_max_scaler = preprocessing.MinMaxScaler()


# plot svi 5, svi 15 and svi 30 in 1 plot
df_svi = pd.read_excel(output_file_location / 'SVI_en_DS_AT.xlsx')
df_svi['year'] = df_svi['datum'].dt.year
df_svi['week'] = df_svi['datum'].dt.week
df_svi = df_svi.set_index('datum')
df_svi = df_svi.fillna(0)

df_bacteria = pd.read_excel(output_file_location / 'bact_samples.xlsx')
df_bacteria = df_bacteria.set_index('datum')

#df_bacteria_filtered = df_bacteria[df_bacteria['Parameter'] == 'Desulfofustis']
df_bacteria_filtered = df_bacteria[df_bacteria['Parameter'] == 'Marinilabilia']


np_scaled = min_max_scaler.fit_transform(df_bacteria_filtered[['Result_Log_Fraction']])
df_bacteria_normalized = pd.DataFrame(np_scaled)
df_bacteria_normalized.index = df_bacteria_filtered.index
df_bacteria_normalized.plot()


np_scaled = min_max_scaler.fit_transform(df_svi[['svi 5', 'svi 15', 'svi 30']])
df_svi_normalized = pd.DataFrame(np_scaled)
df_svi_normalized.index = df_svi.index
df_svi_normalized.plot()

# plot the two graphs
ax = df_svi_normalized.plot()
df_bacteria_normalized.plot(ax=ax)

#df_svi[['svi 5', 'svi 15', 'svi 30']].plot()

df_bacteria_filtered [['Result_Log_Fraction']].plot()

# what dates have the largest svi 30?
print(df_svi.nlargest(15, 'svi 30'))



