from preprocess import get_input_data, de_duplication, noise_remover, translate_to_en
from embeddings import get_tfidf_embd
from modelling.modelling import model_predict, chained_multi_output, hierarchical_modelling
from Common.Data_Model import Data
from Config import Config
import numpy as np
import random

seed = Config.SEED
random.seed(seed)
np.random.seed(seed)

def load_data():
    return get_input_data()

def preprocess_data(df):
    df = de_duplication(df)
    df = noise_remover(df)
    df[Config.TICKET_SUMMARY] = translate_to_en(df[Config.TICKET_SUMMARY].tolist())
    return df

def get_embeddings(df):
    X = get_tfidf_embd(df)
    return X, df

def get_data_object(X, df, target_col=None):
    return Data(X, df, target_col=target_col)

def perform_modelling(data, df, name):
    model_predict(data, df, name)

if __name__ == '__main__':
    print("Loading data")
    df = load_data()
    print(f"Loaded {len(df)} records")
    print("Preprocessing")
    df = preprocess_data(df)
    df[Config.INTERACTION_CONTENT] = df[Config.INTERACTION_CONTENT].values.astype('U')
    df[Config.TICKET_SUMMARY] = df[Config.TICKET_SUMMARY].values.astype('U')
    print("Generating embeddings")
    X, df = get_embeddings(df)
    print(f"Embedding shape: {X.shape}")
    print("\nBaseline: Type 2 only")
    data = get_data_object(X, df, target_col=Config.CLASS_COL)
    perform_modelling(data, df, 'RandomForest - Type 2 (Baseline)')
    print("\nRunning Design Choice 1")
    chained_multi_output(X, df)
    print("\nRunning Design Choice 2")
    hierarchical_modelling(X, df)
    print("\nAll classifications complete.")
