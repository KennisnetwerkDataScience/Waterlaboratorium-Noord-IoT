# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 11:14:28 2018

@author: Pieter

Prepares the data in the kwaliteits data excel sheet

"""

import pandas as pd
from pathlib import Path

# change these
file_location = Path(r'E:\Files\wln')
output_file_location  = Path(r'E:\Files\wln\output')

def cleanup_value(x: str):
    if isinstance(x, str):
        return x.replace('<', '').strip()
    else:
        return x

def prepare_data(d_sheets):
    # lower case
    for df_sheet in d_sheets.values():
        df_sheet.columns = [column.lower().strip() for column in df_sheet.columns]

    # replace the '<' values
    # as some columns this character in them. replace those and make it numeric.
    for df_sheet in d_sheets.values():
        for column in df_sheet.columns:

            if df_sheet[column].dtype == 'O':
                df_sheet[column] = pd.to_numeric(df_sheet[column].map(lambda x: cleanup_value(x)))

    # add date and time columns to online data
    df_online = d_sheets['online data']
    df_online['datum'] = df_online['timestamp'].dt.date
    df_online['datum'] = df_online['datum'].astype('datetime64[ns]')
    df_online['time'] = df_online['timestamp'].dt.time

    return  d_sheets
    # set values with < to value

# read in all sheets from this excel.
# used the ExcelFile class so that it is easy to skip a sheet (like redox AT)
def read_in_all_sheets(file_name):
    result = dict()

    xls = pd.ExcelFile(file_name)

    for sheetname in xls.sheet_names:
        # redox AT is not in a format we can easily read in, so skip it for now
        if sheetname != 'redox AT':
            result[sheetname] = xls.parse(sheetname)

    return result

sheets_kwaliteits_data = read_in_all_sheets(file_location / 'Kennisnetwerk_procesdata_kwaliteitsdata_WLN.xlsx')

sheets_kwaliteits_data  = prepare_data(sheets_kwaliteits_data)

#sheets = list(sheets_kwaliteits_data.keys())


# might be handy to have
# generate a date dataframe (month, year, quarter etc)
def generate_dates(start_year, end_year):

    idx_dates = pd.PeriodIndex(start=start_year, end=end_year, freq='D')
    df_dates = pd.DataFrame(idx_dates.to_timestamp(), columns=['Date'])
    df_dates['datum'] = df_dates['Date']
    df_dates['date_number'] = df_dates['Date'].dt.strftime('%Y%m%d').astype(int)
    df_dates['yearmonth_number'] = df_dates['Date'].dt.strftime('%Y%m').astype(int)
    df_dates['year'] = df_dates['Date'].dt.year
    df_dates['month'] = df_dates['Date'].dt.month
    df_dates['day'] = df_dates['Date'].dt.day
    df_dates['week'] = df_dates['Date'].dt.week
    df_dates['day_of_week'] = df_dates['Date'].dt.weekday

    return df_dates


# merge all dataframes and link with the date column
def merge_dataframes(d_dfs):
    df_result = generate_dates(2014, 2019)

    for _, df_current in d_dfs.items():
        df_result = pd.merge(df_result, df_current, how='left', on='datum')
    return df_result

# also make a daily summary of the online data df
def avg_online_data(df):
    df = df.drop(columns = ['timestamp'])
    df = df.groupby(by=['datum']).mean()
    df['datum'] = df.index
    return df

                  
df_dates = generate_dates(2014, 2019)

df_online_daily = sheets_kwaliteits_data['online data']
df_online_daily  = avg_online_data(df_online_daily)
df_online_daily.to_excel(output_file_location / 'online_daily_summary.xlsx')

df_dates.to_excel(output_file_location / 'dates_period.xlsx')

for key in sheets_kwaliteits_data:
    df_to_save = sheets_kwaliteits_data[key]
    fn = key.replace(' ', '_') + '.xlsx'
    df_to_save.to_excel(output_file_location / fn)
