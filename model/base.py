from typing import Any, Dict
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np

class BaseModel(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def train(self, data):
        pass

    @abstractmethod
    def predict(self, X_test)->np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def print_results(self, y_test, Y_pred)->None:
        raise NotImplementedError

    @abstractmethod
    def data_transform(self):
        pass

    def build (self, values: dict[str, Any]={}):
        if not isinstance(values, dict):
            values = {}
        for key,values in getattr(self, 'defaults', {}).items():
            setattr(self,key,values)
        for key, value in values.items():
            setattr(self, key, value)
