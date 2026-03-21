import numpy as np
import pandas as pd
from model.base import BaseModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from Config import Config
import random

seed = Config.SEED
np.random.seed(seed)
random.seed(seed)

class RandomForest(BaseModel):
    def __init__(self, model_name, embeddings, y):
        super(RandomForest, self).__init__()
        self.model_name = model_name
        self.embeddings = embeddings
        self.y = y
        self.mdl = RandomForestClassifier(n_estimators=1000, random_state=seed, class_weight='balanced_subsample')
        self.predictions = None
        self.data_transform()

    def train(self, data):
        self.mdl = self.mdl.fit(data.X_train, data.y_train)

    def predict(self, X_test):
        predictions = self.mdl.predict(X_test)
        self.predictions = predictions
        return predictions

    def print_results(self, data):
        print(f"\n{'='*60}")
        print(f"Classification Report for: {self.model_name}")
        print(f"{'='*60}")
        print(classification_report(data.y_test, self.predictions, zero_division=0))

    def data_transform(self):
        pass
