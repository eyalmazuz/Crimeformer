import argparse
import os
from typing import Dict, List, Tuple, Union

from model import CrimeModel

import numpy as np
import tensorflow as tf
from tensorflow_addons.metrics import F1Score 
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
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_path', type=str, help='Path of the dataset to load.')
    parser.add_argument('--save_path', type=str, help='Path to save the model.')
    parser.add_argument('--city', type=str, choices=['newyork', 'chicago'], help='Name of the city to run experiments on.')
    parser.add_argument('--year', type=int, choices=[2014, 2015], help='Year of the data to use.')
    parser.add_argument('--data_type', type=str, help='Type of data to train with.')
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
        tests[month] = np.load(month_path)

    return train, tests

def train_model(model_type: str,
                train: np.lib.npyio.NpzFile,
                test: np.lib.npyio.NpzFile,
                crime: str,
                test_size: float=0.2) -> None:

    x, y = train['x'], train['y']
    
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=test_size, shuffle=False, random_state=42)
    
    weight_ratio = float(len(y_train[y_train == 1])) / float(len(y_train[y_train == 0]))

    if model_type == 'xgboost':
        model = XGBClassifier(n_estimators=50, objective='binary:logistic', n_jobs=-1, scale_pos_weight=weight_ratio)

        model.fit(x_train, y_train)

        proba = model.predict_proba(x_val)
        preds = model.predict(x_val)

    else:
        model = CrimeModel(32)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.AUC()])

        model.fit(x_train, y_train, epochs=5, batch_size=64, verbose=1, validation_data=(x_val, y_val), class_weight={0: weight_ratio, 1: 1})
        proba = model.predict(x_test, y_test, batch_size=64)
        preds = [1 if prob > 0.5 else 0 for prob in proba]
        
    print(f'{crime}: F1: {round(f1_score(y_val, preds), 4)} AUC: {round(roc_auc_score(y_val, proba[:, 1]), 4)}')

def main():

    parser = arg_parse()

    if parser.model == 'xgboost':
        path = f'{parser.load_path}/historic/{parser.year}'

    if parser.model == 'tensorflow':
        path = f'{parser.load_path}/time_series/{parser.year}'

    for crime in os.listdir(path):
        crime_path = f'{path}/{crime}'
        train, tests = load_data(crime_path)
        train_model(parser.model, train, tests, crime, parser.test_size)


if __name__ == "__main__":
    main()