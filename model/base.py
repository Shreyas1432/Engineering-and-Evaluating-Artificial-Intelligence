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

    def build(self, values=dict[str, any]={}):
        values = values if isinstance(values, dict) else {}
        if hasattr(self, 'defaults'):
            self.__dict__.update(self.defaults)
        self.__dict__.update(values)
        return self
