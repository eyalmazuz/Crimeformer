import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from typing import Dict, List, Tuple, Union
from datetime import datetime

from model import CrimeModel

import numpy as np
import tensorflow as tf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import roc_auc_score, f1_score

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_path', type=str, help='Path of the dataset to load.')
    parser.add_argument('--save_path', type=str, help='Path to save the results.')
    parser.add_argument('--window_size', type=int, default=10, help='The size of the window to use for experiments.')
    parser.add_argument('--year', type=int, choices=[2014, 2015], help='Year of the data to use.')
    parser.add_argument('--model', type=str, help='Models to train choices are xgboost or tensorflow LSTM.')
    parser.add_argument('--test_size', type=float, default=0.2, help='Amount of the to give the test set. Ranges between 0.01-0.99 .')


    return parser.parse_args()


def load_data(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads the data from path for a spesific city.
    The type of the data to load is based on the model used.

    Parameters
    ----------
    path: str
        Path of the data.

    Returns
    -------
    X: np.ndarray
        Numpy array of inputs for the model.

    y: np.ndarray
        Numpy array of labels.
    """

    train = np.load(f'{path}/train.npz')
    tests = {}
    for month in os.listdir(f'{path}/test'):
        month_path = f'{path}/test/{month}'
        tests[month.split('.')[0]] = np.load(month_path)

    return train, tests

def train_model(model_type: str,
                train: np.lib.npyio.NpzFile,
                test: np.lib.npyio.NpzFile,
                crime: str,
                data_type: str,
                test_size: float=0.2) -> None:

    x, y = train['x'], train['y']
    
#     x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=test_size, shuffle=False, random_state=42)
    
    weight_ratio = float(len(y[y == 1])) / float(len(y[y == 0]))

    results = pd.DataFrame({'crime': [],'model': [], 'data_type': [], 'month': [],'f1': [], 'auc': []})
    
    if model_type == 'xgboost':
        model = XGBClassifier(n_estimators=200, objective='binary:logistic',
                              n_jobs=-1,
                              eval_metric='logloss')

        model.fit(x, y)
        
        for month in test:
            x_test, y_test = test[month]['x'], test[month]['y']
            proba = model.predict_proba(x_test)[:, 1]
            preds = model.predict(x_test)
            
            auc = round(roc_auc_score(y_test, proba), 4)
            f1 = round(f1_score(y_test, preds), 4)
            
            results = results.append(pd.Series({'crime': crime,
                                      'model': model_type,
                                      'data_type':  data_type,
                                      'month': month,
                                      'f1': f1,
                                      'auc': auc}), ignore_index=True)

    else:
        model = CrimeModel(32)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.AUC()])

        model.fit(x, y, epochs=5, batch_size=64, verbose=0)
        
        for month in test:
            x_test, y_test = test[month]['x'], test[month]['y']
            proba = model.predict(x_test, batch_size=64)
            preds = [1 if prob > 0.5 else 0 for prob in proba]
            
            auc = round(roc_auc_score(y_test, proba), 4)
            f1 = round(f1_score(y_test, preds), 4)
            
            results = results.append(pd.Series({'crime': crime,
                                      'model': model_type,
                                      'data_type':  data_type,
                                      'month': month,
                                      'f1': f1,
                                      'auc': auc}), ignore_index=True)
        
        
    return results
    

def main():

    parser = arg_parse()

    for model in tqdm(['tensorflow', 'xgboost']):

        df = pd.DataFrame({'crime': [],'model': [], 'month': [],'f1': [], 'auc': []})

        for data_type in tqdm(['historic', 'embedding'], leave=False):
            
            if model == 'xgboost':
                path = f'{parser.load_path}/{parser.window_size}/{data_type}/regular/{parser.year}'

            elif model == 'tensorflow':
                path = f'{parser.load_path}/{parser.window_size}/{data_type}/time_series/{parser.year}'
            
            for crime in tqdm(os.listdir(path), leave=False):
                crime_path = f'{path}/{crime}'
                train, tests = load_data(crime_path)
                results = train_model(model, train, tests, crime, data_type, parser.test_size)
                df = df.append(results)
        
        for metric in ['auc', 'f1']:
            g = sns.FacetGrid(df, col="crime", height=5, aspect=.8)
            ax = g.map_dataframe(sns.barplot, x="month", y=metric, hue="data_type", palette=sns.color_palette("tab10"))
            plt.subplots_adjust(top=0.8)
            g.add_legend()
            g.fig.suptitle(f'Year: {parser.year}')
            ax.savefig(f"{parser.save_path}/{model}_{parser.window_size}_{metric}_{str(datetime.now())}.png")


if __name__ == "__main__":
    main()