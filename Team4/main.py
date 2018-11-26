# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 21:15:46 2018
@author: pahuizinga

Doel:
Achterhaal de oorzaak van de SVI pieken
of
ontwikkel een model die de pieken kan zien aankomen.
"""
# change working directory
#import os
#path = r'G:\Python\WLN'
#os.chdir(path)

import pandas as pd
import numpy as np
import config as config
import basis_data_controls as bdc
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
#from sklearn.metrics import precision_recall_fscore_support
#import datetime
from sklearn.metrics import confusion_matrix
import seaborn as sb
#%%
## =============================================================================
# LOAD DATA
# verwijder eerst (handmatig) sheet redox!!
proces_data = pd.read_excel(config.procesdata, sheetname=None) # sheetname=None: read all sheets from the Excel file
proces_data['online data'].rename(columns={'Timestamp': 'datum'}, inplace=True)
#%%
# =============================================================================
# BASIC STATISTICS

#%%
# =============================================================================
# DATA PREP
def prepare_data(dictionairy):

    for key in dictionairy.keys():
        # lower case column headers
        dictionairy[key].columns = map(str.lower, dictionairy[key].columns)
        # strip column names.
        
        dictionairy[key].columns = dictionairy[key].columns.str.strip()
        # remove non-string characters from column names
        
        # replace spaces with underscore
        
        # group df online data by date and calculate average per day
            
prepare_data(proces_data)
#%%
def interpolate_svi(df):
    mindate = df['datum'].min()
    maxdate = df['datum'].max()
    
    x = pd.to_datetime(df['datum'])
    
    xnew = pd.to_datetime(pd.date_range(mindate,maxdate, freq='D')).to_series()
    xnew.reset_index(drop=True, inplace=True)
    xnew.name = 'datum'
    dfnew = pd.DataFrame(xnew)
    
    for column in df.columns.tolist():
        if not column == 'datum':
            y = df[column]
            ynew = np.interp(pd.to_numeric(xnew), pd.to_numeric(x), y)
            ynew = pd.Series(ynew)
            ynew.name = column
            ynew = pd.DataFrame(ynew)
            dfnew = pd.concat([dfnew, ynew], axis = 1)
    
    return dfnew
#%%
# =============================================================================
# COMBINE ALL DATA
def merge_dataframes(dictionairy):
    all_stats1 = bdc.summarize_statistics(proces_data)
    # achterhaal min en max datums van alle data
    min_date = all_stats1[all_stats1['column']=='datum'].loc[:,'min'].min()
    max_date = all_stats1[all_stats1['column']=='datum'].loc[:,'max'].max()
    # maak datum dataframe
    df_total = pd.DataFrame(pd.period_range(start=min_date, end=max_date, freq='D').to_timestamp(), columns={'datum'})
    df_total['datum'] = pd.to_datetime(df_total['datum']).dt.date
    
    for key in dictionairy.keys():
        if key == 'online data':
            df_tmp = dictionairy[key]
            df_tmp['datum'] = pd.to_datetime(df_tmp['datum']).dt.date
            df_tmp = df_tmp.groupby(by=['datum']).mean()
            df_tmp['datum'] = df_tmp.index
        elif key == 'SVI en DS AT':
            df_tmp = interpolate_svi(dictionairy[key])       
            df_tmp['datum'] = pd.to_datetime(df_tmp['datum']).dt.date
        else:
            df_tmp = dictionairy[key]
            df_tmp['datum'] = pd.to_datetime(df_tmp['datum']).dt.date

        df_total = pd.merge(df_total, df_tmp, how='outer', on='datum') 
        
    return df_total
    
merged_data = merge_dataframes(proces_data)
merged_data['datum'] = pd.to_datetime(merged_data['datum'])

all_stats = bdc.BasicStatistics(merged_data)
#%%
# data cleansing
# optie 1: verwijder alle records waar svi 30 geen waarde heeft
# verwijder vervolgens alle kolommen die niet meer dan x% gevuld zijn.
max_nan = .25 #if nan% > x%: exclude
totalrecords = merged_data.shape[0] 

# verwijder kolommen
dataset1 = merged_data.dropna(subset=['svi 30'])
totalrecords = dataset1.shape[0] 

## vervang nul waarden met nan
cols = dataset1.columns.tolist() 
dataset1[cols] = dataset1[cols].replace({'0':np.nan, 0:np.nan})

for x in dataset1.columns.tolist():
    if (dataset1[x].isnull().sum()/totalrecords) > max_nan:
        del dataset1[x]
        
# handle na's
# niet heel mooi op deze manier....
dataset1.fillna(0, inplace=True)

# optie 2: verwijder all records voor en na de eerste en laatste datum van svi 30.
# interpoleer de lege waarden van svi 30
# verwijder vervolgens alle kolommen die niet meer dan x% gevuld zijn.
#%%
# maak classifier column
threshold = 60  # svi 30 grens: alles hierboven lijkt 'niet goed'
days =20 # aantal dagen dat terug wordt gekeken om stijging van de svi 30 curve te bepalen
# default waarde is 0
dataset1['class'] = 0

for record in range(5, dataset1.shape[0]):
    if dataset1['svi 30'].iloc[record] >= threshold and dataset1['svi 30'].iloc[record - days] < dataset1['svi 30'].iloc[record]:
        dataset1['class'].iloc[record] = 1
#%%
#plot
x = dataset1['datum']
y = dataset1['svi 30'] 
z = dataset1['class']

fig, ax1 = plt.subplots()
fig.set_figheight(8)
fig.set_figwidth(15)

color = 'tab:red'
ax1.set_xlabel('date')
ax1.set_ylabel('svi 30', color=color)
ax1.plot(x, y, color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  

color = 'tab:green'
ax2.set_ylabel('class', color=color) 
ax2.plot(x, z, color=color)
ax2.tick_params(axis='class', labelcolor=color)
ax2.yticks(np.arange(0, 1, step=1))

fig.tight_layout() 
plt.savefig('output/svi_classifier.png')
plt.show()

#%%    
#verwijder svi kolommen
for nr in range(5, 35, 5):
    del dataset1[str('svi '+ str(nr))]
#%%
# STATISTICS
all_stats = bdc.BasicStatistics(dataset1)        
print(all_stats)
#%%
# test train split
from sklearn.model_selection import train_test_split
features = dataset1.iloc[:,1:-1]
classes = dataset1.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(features, classes, \
                            stratify=classes, random_state=0, train_size=.75, test_size=.25) 

#%%
# LOGISTIC REGRESSION
algo = 'Logistic Regression'

cross_validation = 10

C_range = [3, 4, 5, 6, 7, 8, 9, 10]
penalty_range = ['l1','l2']
tolerance_range = [0.000001, 0.00001, 0.0001]   
solver_range = ['liblinear']#,'lbfgs','newton-cg','sag']
cw_range = ['balanced']

param_grid = [{'lr__C': C_range
               ,'lr__penalty': penalty_range
               ,'lr__tol': tolerance_range
               ,'lr__class_weight': cw_range
               ,'lr__solver': solver_range
               }]

pipe_lr = Pipeline(steps=[
    ('pca', PCA()),
    ('lr', LogisticRegression())
])

classifier = GridSearchCV(estimator=pipe_lr, param_grid=param_grid, cv=cross_validation)
    
classifier.fit(X_train,y_train)

print('best parameters')
print(classifier.best_params_)

predictions = classifier.predict(features)

print(confusion_matrix(classes, predictions))
#run_output(X_train, y_train, algo, 'train')
print(classifier.score(features,classes))

#%%
# =============================================================================
# OUTPUT
all_stats.to_html('output/statistics.html')
dataset1.to_csv('output/dataset1.csv')
    
    
# =============================================================================    
## correlations

corr = features.corr()

mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
cmap = sb.diverging_palette(240, 10, as_cmap=True)
fig, ax = plt.subplots(figsize=(20,20))  
sb.heatmap(corr, mask=mask, cmap=cmap, ax=ax, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.savefig('output/corr_matr.png', dpi=300)



#%%
#%%
#df_out = pd.DataFrame(columns = ['datetime', 'algorithm', 'dataset', 'train_size', 'max_nan', 'cross validation', 'score', 'tn', 'fp', 'fn', 'tp', 'precision', 'recall', 'ben_prec', 'ben_recall', 'ben_f1', 'ben_sup', 'path_prec', 'path_recall', 'path_f1', 'path_support'])
#
#def add_result(algo, dataset, train_size, max_nan,cv,mcp,score,tn,fp,fn,tp,ben_prec,ben_recall,ben_f1,ben_sup,path_prec,path_recall,path_f1,path_support):
#    df_out.loc[len(df_out)+1] = [datetime.datetime.now(), algo, dataset, train_size, max_nan, cv, mcp, score, tn, fp, fn, tp, (tp/(tp+fp)), (tp/(tp+fn)),ben_prec, ben_recall, ben_f1,ben_sup,path_prec,path_recall,path_f1,path_support]
#
#
#def run_output(featureset, classset, algo, dataset):
#    predictions = classifier.predict(featureset)
##    score = classifier.score(featureset,classset)
#    print(confusion_matrix(classset, predictions))
#    tn, fp, fn, tp = confusion_matrix(classset, predictions).ravel()
#    ben_prec, ben_recall, ben_f1,ben_sup,path_prec,path_recall,path_f1,path_support = np.array(precision_recall_fscore_support(classset, predictions)).T.ravel()
#    print(np.array(precision_recall_fscore_support(classset, predictions)).T.ravel())
#    #add_result(algo,dataset, train, max_nan, cross_validation, score, tn, fp, fn, tp,ben_prec, ben_recall, ben_f1,ben_sup,path_prec,path_recall,path_f1,path_support)
# 

