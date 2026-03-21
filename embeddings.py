import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from Config import Config

def get_tfidf_embd(df):
    combined_text = (
        df[Config.TICKET_SUMMARY].fillna('').astype(str) + ' ' +
        df[Config.INTERACTION_CONTENT].fillna('').astype(str)
    )
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2), min_df=2, max_df=0.95)
    X = vectorizer.fit_transform(combined_text).toarray()
    return X
