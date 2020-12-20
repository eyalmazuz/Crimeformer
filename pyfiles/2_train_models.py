import argparse
from typing import Dict, List, Tuple, Union

from model import CrimeModel

import numpy as np
import tensorflow as tf
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
    parser.add_argument('--save_path', type=str, help='Path to save the npy files.')
    parser.add_argument('--city', type=str, choices=['newyork', 'chicago'], help='Name of the city to run experiments on.')
    parser.add_argument('--model', type=str, help='Models to train choices are xgboost or tensorflow LSTM.')
    parser.add_argument('--test_size', type=float, default=0.2, help='Amount of the to give the test set. Ranges between 0.01-0.99 .')


    return parser.parse_args()


def load_data(path: str, city: str, model: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads the data from path for a spesific city.
    The type of the data to load is based on the model used.

    Parameters
    ----------
    path: str
        Path of the data.

    city: str
        Name of the city to predict on.
    
    model: str
        Type of model to run data for. Determines the type of data to load

    
    Returns
    -------
    X: np.ndarray
        Numpy array of inputs for the model.

    y: np.ndarray
        Numpy array of labels.
    """

    if model == 'tensorflow':
        full_path = f'{path}/{city}_time_series_data.npz'

    if model == 'xgboost':
        full_path = f'{path}/{city}_regular_data.npz'

    data = np.load(full_path)
    return data['x'], data['y']

def train_model(model_type: str,
                train: Tuple[np.ndarray, np.ndarray],
                test: Tuple[np.ndarray, np.ndarray]) -> None:

    x_train, y_train = train
    x_test, y_test = test
    weight_ratio = float(len(y_train[y_train == 1])) / float(len(y_train[y_train == 0]))

    if model_type == 'xgboost':
        model = XGBClassifier(n_estimators=50, objective='binary:logistic', n_jobs=-1, scale_pos_weight=weight_ratio)

        model.fit(x_train, y_train)

        proba = model.predict_proba(x_test)
        preds = model.predict(x_test)

        print(f'F1: {round(f1_score(y_test, preds), 4)} AUC: {round(roc_auc_score(y_test, proba[:, 1]), 4)}')

    else:
        model = CrimeModel(32)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC()])

        model.fit(x_train, y_train, epochs=5, batch_size=64, verbose=1, validation_data=(x_test, y_test), class_weight={0: weight_ratio, 1: 1})


def main():

    parser = arg_parse()

    X, y = load_data(parser.load_path, parser.city, parser.model)
    print(f'{X.shape=}, {y.shape=}')

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=parser.test_size, shuffle=False)
    
    train_model(parser.model, (x_train, y_train), (x_test, y_test))


if __name__ == "__main__":
    main()