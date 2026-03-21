import pandas as pd
import numpy as np
from Config import Config

def remove_low_frequency_classes(df, column, min_count=None):
    if min_count is None:
        min_count = Config.MIN_CLASS_COUNT
    class_counts = df[column].value_counts()
    valid_classes = class_counts[class_counts >= min_count].index
    return df[df[column].isin(valid_classes)]

def remove_nan_rows(df, column):
    return df.dropna(subset=[column])
