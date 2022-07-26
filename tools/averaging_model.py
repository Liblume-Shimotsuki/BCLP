from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
import numpy as np
import pandas as pd
from copy import deepcopy
from sklearn.metrics import mean_absolute_percentage_error
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras import regularizers


class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models

    def fit(self, X, y):
        self.models_ = [deepcopy(x) for x in self.models]
        for model in self.models_:
            model.fit(X, y)
        return self

    def predict(self, X):
        predictions = np.column_stack([
            model.predict(X) for model in self.models_
        ])
        return np.mean(predictions, axis=1)


class OptionalModel(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, base_model, log_target) -> None:
        super().__init__()
        self.base_model = base_model
        self.log_target = log_target

    def fit(self, X, y):
        features_input = X
        target = np.log(y) if self.log_target else y
        self.base_model.fit(features_input, target)
        return self

    def predict(self, X):
        pred = self.base_model.predict(X)
        return np.exp(pred) if self.log_target else pred

    def eval(self, X, y):
        pred = self.predict(X)
        return mean_absolute_percentage_error(y, pred) * 100


class OptionalNnModels(OptionalModel):
    def __init__(self, base_model, target_scaler) -> None:
        super().__init__(base_model, log_target=False)
        self.target_scaler = target_scaler

    def fit(self, X, y):
        y = y.copy()
        if isinstance(y, np.ndarray):
            target = self.target_scaler.transform(y.reshape(-1, 1))
        elif isinstance(y, pd.Series):
            target = self.target_scaler.transform(y.values.reshape(-1, 1))
        else:
            raise TypeError("y must be a numpy array or pandas series")
        return super().fit(X, target)

    def predict(self, X):
        pred = super().predict(X)
        return self.target_scaler.inverse_transform(pred)


def build_nn():
    model = Sequential()
    model.add(Dense(units=128, activation='relu',
              kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4), input_dim=6))
    model.add(Dense(units=32, activation='relu',
              kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4)))
    model.add(Dense(units=1, activation='linear'))
    model.compile(loss='mse', optimizer='adam')
    return model
