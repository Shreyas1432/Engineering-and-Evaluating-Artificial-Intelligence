import random
import numpy as np
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC
from Config import Config
from model.base import BaseModel
seed = Config.SEED
np.random.seed(seed)
random.seed(seed)
class LinearSVM(BaseModel):
    def __init__(self, model_name, embeddings, y):
        super(LinearSVM, self).__init__()
        self.model_name = model_name
        self.embeddings = embeddings
        self.y = y
        self.mdl = LinearSVC(random_state=seed, class_weight='balanced', max_iter=5000)
        self.predictions = None
        self.data_transform()
    def train(self, data):
        self.mdl = self.mdl.fit(data.X_train, data.y_train)
    def predict(self, X_test) -> np.ndarray:
        return self.mdl.predict(X_test)
    def print_results(self, y_test, Y_pred):
        print(f"Classification Report for: {self.model_name}")
        print(classification_report(y_test, Y_pred, zero_division=0))
    def data_transform(self):
        pass