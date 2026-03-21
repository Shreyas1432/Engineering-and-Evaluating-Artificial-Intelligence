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
    def predict(self, X_test):
        pass

    @abstractmethod
    def print_results(self, data):
        pass

    @abstractmethod
    def data_transform(self):
        pass

    def build(self, values={}):
        values = values if isinstance(values, dict) else {}
        if hasattr(self, 'defaults'):
            self.__dict__.update(self.defaults)
        self.__dict__.update(values)
        return self
