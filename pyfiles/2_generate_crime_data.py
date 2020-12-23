import os
import argparse
import math
import json
import dateutil.parser
from typing import List, Dict, Tuple

import pandas as pd
import numpy as np
from tqdm import tqdm
tqdm.pandas()

TestData = Dict[int, Dict[str, np.ndarray]]
Data = Dict[str, np.ndarray]

AREA_COLUMN = 'ADDR_PCT_CD'
DATE_COLUMN = 'CMPLNT_FR_DT'
OFFENCE_CODE = 'KY_CD'
OFFENCE_DESC = 'OFNS_DESC'
CB = 'BORO_NM'

LOAD_COLUMNS = [AREA_COLUMN, DATE_COLUMN, OFFENCE_CODE, OFFENCE_DESC, CB]
LOWEST_DATE = dateutil.parser.parse('1/1/2000').date()

crime_types = None




embedding_dict = None

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_path', type=str, help='Path of the dataset to load.')
    parser.add_argument('--crime_code_path', type=str, help='Path of the crime code to load.')
    parser.add_argument('--save_path', type=str, help='Path to save the npy files.')
    parser.add_argument('--window_size', type=int, default=5, help='Size of the window.')   

    return parser.parse_args()

def date_from_str(date: str):
    """
    Converts string to datetime object.
    If date is lower than thershold we return nan.

    Parameters
    ----------
    date: str
        String of the date.

    Returns
    -------
    date: DateTime
        DateTime object of the string that was given.

    """
    if isinstance(date, float) and math.isnan(date):
        return np.nan
    date = dateutil.parser.parse(date).date()
    if date < LOWEST_DATE:
        return np.nan
    return date

def generate_test_data(df: pd.DataFrame, start_date, end_date, window=5, data_type='regular', use_embedding: bool=False) -> TestData:
    """
    Generates trainning data for each month in the test set.
    Each month contains all the training data for all precints.

    Parameters:
    -----------
    df: pd.DataFrame
        dataframe which contains crime data for a specific crime for a specific year.

    start_date: DateTime
        DateTime object of the date which the test set starts at.

    end_date: DateTime
        DateTime object of the date which the test set ends at.
        
    window: int
        Window size for look back.

    data_type: str
        String representing the type of data to generate.

    use_embedding: bool
        If to incorporate embeddings in the data.


    Returns
    -------
    data: Data
        Dictionary that contains mapping from months to test data.
    """
    if data_type == 'regular':
        columns = 'has_crime'
    else:
        columns = ['has_crime', 'area']

    data = {}
    for area in df[AREA_COLUMN].unique():
        
        area_df = df[df[AREA_COLUMN] == area]
        area_df = area_df.sort_values(DATE_COLUMN, ascending=True)
        crime_dates = area_df[DATE_COLUMN].unique()

        borough = area_df[CB].unique().tolist()[0].lower()

        dates = pd.date_range(start_date, end_date, freq='d')
        has_crime = [1 if date in crime_dates else 0 for date in dates]
        area_crime_df = pd.DataFrame({'date': dates.date,
                                      'has_crime': has_crime,
                                      'month': dates.month,
                                      'area': area }) 
        
        for month in area_crime_df['month'].unique(): 
        
            if month not in data:
                data[month] = {'x': [], 'y': []}
            
            month_df = area_crime_df[area_crime_df['month'] == month]
            x = []
            y = []
            for i in range(window, month_df.shape[0]):
                if data_type == 'regular':
                    instance = np.array(month_df[i - window : i][columns].tolist() + month_df['area'].unique().tolist())
                    if use_embedding:
                        instance = np.hstack([instance, embedding_dict[borough]])
                else:
                    instance = month_df[i - window : i][columns].values
                    if use_embedding:
                        embs = np.array([embedding_dict[borough]] * window)
                        instance = np.hstack([instance, embs]) 

                x.append(instance)
                y.append(month_df.iloc[i]['has_crime'])

            data[month]['x'].append(x)
            data[month]['y'].append(y)
        
    for month in data:
        data[month]['x'] = np.vstack(data[month]['x'])
        data[month]['y'] = np.hstack(data[month]['y'])

    return data

def generate_train_data(df: pd.DataFrame, start_date, end_date, window=5, data_type='regular', use_embedding: bool=False) -> Data:
    """
    Generates trainning data for each month in the test set.
    Each month contains all the training data for all precints.

    Parameters:
    -----------
    df: pd.DataFrame
        dataframe which contains crime data for a specific crime for a specific year.

    start_date: DateTime
        DateTime object of the date which the train/validation set starts at.

    end_date: DateTime
        DateTime object of the date which the train/validation set ends at.
        
    window: int
        Window size for look back.

    data_type: str
        String representing the type of data to generate.

    use_embedding: bool
        If to incorporate embeddings in the data.

    Returns
    -------
    data: Data
        Dictionary that contains mapping from x/y numpy arrays.
    """

    if data_type == 'regular':
        columns = 'has_crime'
    else:
        columns = ['has_crime', 'area']

    data = {'x': [], 'y': []}
    
    for area in df[AREA_COLUMN].unique():
        
        area_df = df[df[AREA_COLUMN] == area]
        area_df = area_df.sort_values(DATE_COLUMN, ascending=True)
        crime_dates = area_df[DATE_COLUMN].unique()

        borough = area_df[CB].unique().tolist()[0].lower()

        dates = pd.date_range(start_date, end_date, freq='d')
        has_crime = [1 if date in crime_dates else 0 for date in dates]
        area_crime_df = pd.DataFrame({'date': dates.date,
                                      'has_crime': has_crime,
                                      'month': dates.month,
                                      'area': area }) 
        x = []
        y = []

        for i in range(window, area_crime_df.shape[0]):
            if data_type == 'regular':
                instance = np.array(area_crime_df[i - window : i][columns].tolist() + area_crime_df['area'].unique().tolist())
                if use_embedding:
                    instance = np.hstack([instance, embedding_dict[borough]])
            else:
                instance = area_crime_df[i - window : i][columns].values
                if use_embedding:
                    embs = np.array([embedding_dict[borough]] * window)
                    instance = np.hstack([instance, embs]) 
            
                
            x.append(instance)
            y.append(area_crime_df.iloc[i]['has_crime'])
        
        data['x'].append(x)
        data['y'].append(np.array(y))
        
    data['x'] = np.vstack(data['x'])
    data['y'] = np.hstack(data['y'])

    return data

def generate_area_data(year_df: pd.DataFrame, year: int, save_path: str, window_size: int=5, data_type: str='regular', use_embedding: bool=False) -> None:
    
    """
    Generates data for a specific area.


    Parameters
    ----------
    year_df: pd.DataFrame
        DataFrame that holds crime data for a specific crime for a specific year.

    year: int 
        Year of the crime data.

    save_path: str
        Path to save the data at.

    window: int
        Window size for look back.

    data_type: str
        String representing the type of data to generate.

    use_embedding: bool
        If to incorporate embeddings in the data.
    """
    train_date = dateutil.parser.parse(f'1/1/{year}').date()
    test_date = dateutil.parser.parse(f'8/1/{year}').date()
    end_date = dateutil.parser.parse(f'12/31/{year}').date()

    train_df = year_df[(year_df[DATE_COLUMN] >= train_date) & (year_df[DATE_COLUMN] < test_date)]
    test_df = year_df[(year_df[DATE_COLUMN] >= test_date)]
    
    train = generate_train_data(train_df, start_date=train_date, end_date=test_date, window=window_size ,data_type=data_type, use_embedding=use_embedding)
    test = generate_test_data(test_df, start_date=test_date, end_date=end_date, window=window_size, data_type=data_type, use_embedding=use_embedding)
    
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    np.savez(f'{save_path}/train.npz', **train)


    for month in test:
        test_month_data = test[month]

        if not os.path.exists(f'{save_path}/test'):
            os.mkdir(f'{save_path}/test')

        np.savez(f'{save_path}/test/{month}.npz', **test_month_data)

def generate_crime_data(year_df: pd.DataFrame, year: int, save_path: str, window_size: int=5, data_type: str='regular', use_embedding=False) -> None:
    """
    Generates data for all crimes in the new york dataset.

    Parameters
    ----------
    df: pd.DataFrame
        new york crime dataframe.

    year: int
        Year.

    save_path: str
        Path to save the data at.

    window: int
        Window size for look back.

    data_type: str
        String representing the type of data to generate.

    use_embedding: bool
        If to incorporate embeddings in the data.
    """
    
    selected_crimes = crime_types[crime_types['OFNS_DESC'].isin(['ROBBERY',
                                           'BURGLARY',
                                           'FELONY ASSAULT',
                                           'GRAND LARCENY',])]['KY_CD'].tolist()

    for crime in tqdm(selected_crimes, leave=False):
        crime_df = year_df[year_df[OFFENCE_CODE] == crime]
        crime_type = crime_types[crime_types[OFFENCE_CODE] == crime][OFFENCE_DESC].iloc[0].replace('/', '_')
        try:
            crime_path = f'{save_path}/{crime_type}'
            if not os.path.exists(crime_path):
                os.mkdir(crime_path)

            generate_area_data(crime_df, year, crime_path, window_size, data_type, use_embedding)
        except ValueError:
            print(f'failed at {crime_type}')

def generate_year_data(df: pd.DataFrame, save_path: str, window_size: int=5, data_type: str='regular', use_embedding: bool=False):
    """
    Generates data for a specific crime for all years in a range.

    Parameters
    ----------
    crime_df: pd.DataFrame
        Dataframe with all the crime data for a specific crime for all years.

    save_path: str
        Path to save the data at.

    window: int
        Window size for look back.

    data_type: str
        String representing the type of data to generate.

    use_embedding: bool
        If to incorporate embeddings in the data.
    """
    for year in tqdm(range(2014, 2016), leave=False):   
        START_DATE = dateutil.parser.parse(f'1/1/{year}').date()
        END_DATE = dateutil.parser.parse(f'1/1/{year + 1}').date()
        year_df = df[(df[DATE_COLUMN] > START_DATE) & (df[DATE_COLUMN] < END_DATE)]
        
        year_path = f'{save_path}/{str(year)}'
        if not os.path.exists(year_path):
                os.mkdir(year_path)

        generate_crime_data(year_df, year, year_path, window_size, data_type, use_embedding)

def main():

    global embedding_dict, crime_types

    parser = arg_parse()

    crime_types = pd.read_csv(parser.crime_code_path)
    crime_types = crime_types[~crime_types[OFFENCE_DESC].isnull()]

    df = pd.read_csv(parser.load_path, usecols=LOAD_COLUMNS)
    df = df[df[AREA_COLUMN] != -99.0]

    df[DATE_COLUMN] = df[DATE_COLUMN].progress_apply(lambda date: date_from_str(date))

    for column in LOAD_COLUMNS:
        df = df[~df[column].isnull()]

    df = df.sort_values(DATE_COLUMN)

    for data_type in tqdm(['regular', 'time_series']):
        for use_embedding in tqdm([True, False], leave=False):
            path = f'{parser.save_path}/{str(parser.window_size)}'

            if not os.path.exists(path):
                os.mkdir(path)
            
            if use_embedding:
                path = f'{path}/embedding/'
                with open('../data/jsons/newyork_borough_emb.json', 'r') as f:
                    embedding_dict = json.load(f)
            else: 
                path = f'{path}/historic/'

            if not os.path.exists(path):
                os.mkdir(path)
            
            path = f'{path}/{data_type}'

            if not os.path.exists(path):
                os.mkdir(path)
            
            generate_year_data(df, path, parser.window_size, data_type, use_embedding)

if __name__ == "__main__":
    main()
