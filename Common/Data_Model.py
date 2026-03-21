import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from Config import Config
from utils import remove_low_frequency_classes, remove_nan_rows
import random

seed = Config.SEED
random.seed(seed)
np.random.seed(seed)

class Data_container:
    def __init__(self, train_x, train_y, test_x, test_y, metadata=None):
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y
        self.metadata = metadata if metadata is not None else {}

    def get_train_data(self):
        return self.train_x, self.train_y

    def get_test_data(self):
        return self.test_x, self.test_y

    def get_shape(self):
        return {
            'train_x': self.train_x.shape if hasattr(self.train_x, 'shape') else len(self.train_x),
            'train_y': len(self.train_y),
            'test_x': self.test_x.shape if hasattr(self.test_x, 'shape') else len(self.test_x),
            'test_y': len(self.test_y)
        }

    def __repr__(self):
        shapes = self.get_shape()
        return (
            f"Data_container(train_x={shapes['train_x']}, train_y={shapes['train_y']}, "
            f"test_x={shapes['test_x']}, test_y={shapes['test_y']})"
        )

class HierarchicalDataContainer:
    def __init__(self, parent_class, level, data_container, children=None):
        self.parent_class = parent_class
        self.level = level
        self.data = data_container
        self.children = children if children is not None else {}

    def add_child(self, class_label, child_container):
        self.children[class_label] = child_container

    def get_child(self, class_label):
        return self.children.get(class_label)

    def get_all_children(self):
        return self.children

    def __repr__(self):
        return (
            f"HierarchicalDataContainer(parent_class='{self.parent_class}', "
            f"level={self.level}, children={len(self.children)})"
        )

class Data(Data_container):
    def __init__(self, X, df, target_col=None):
        if target_col is None:
            target_col = Config.CLASS_COL
        self.target_col = target_col
        valid_mask = df[target_col].notna()
        df_clean = df[valid_mask].reset_index(drop=True)
        X_clean = X[valid_mask.values]
        df_clean = remove_low_frequency_classes(df_clean, target_col)
        valid_indices = df_clean.index.tolist()
        X_clean = X_clean[valid_indices]
        df_clean = df_clean.reset_index(drop=True)
        self.y = df_clean[target_col].values
        self.embeddings = X_clean
        X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
            X_clean, self.y, np.arange(len(df_clean)),
            test_size=0.2, random_state=seed, stratify=self.y
        )
        super().__init__(train_x=X_train, train_y=y_train, test_x=X_test, test_y=y_test,
                         metadata={'target_col': target_col, 'n_classes': len(np.unique(self.y))})
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.train_df = df_clean.iloc[train_idx].reset_index(drop=True)
        self.test_df = df_clean.iloc[test_idx].reset_index(drop=True)

    def get_type(self):
        return self.y

    def get_X_train(self):
        return self.X_train

    def get_X_test(self):
        return self.X_test

    def get_type_y_train(self):
        return self.y_train

    def get_type_y_test(self):
        return self.y_test

    def get_train_df(self):
        return self.train_df

    def get_embeddings(self):
        return self.embeddings

    def get_type_test_df(self):
        return self.test_df
