import pandas as pd
import numpy as np
import re
from Config import Config

def get_input_data():
    dfs = []
    for file_path in Config.DATA_FILES:
        df = pd.read_csv(file_path)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

def de_duplication(df):
    df = df.drop_duplicates(subset=[Config.TICKET_SUMMARY, Config.INTERACTION_CONTENT], keep='first')
    return df.reset_index(drop=True)

def noise_remover(df):
    def clean_text(text):
        if pd.isna(text):
            return text
        text = str(text)
        text = re.sub(r'&amp;', '&', text)
        text = re.sub(r'&lt;', '<', text)
        text = re.sub(r'&gt;', '>', text)
        text = re.sub(r'\S+@\S+', '', text)
        text = re.sub(r'http\S+|www\.\S+', '', text)
        text = re.sub(r'\*+\(PHONE\)', '', text)
        text = re.sub(r'\*+\(PER(SON)?\)', '', text)
        text = re.sub(r'\*+\(LOC\)', '', text)
        text = re.sub(r'[^\w\s.,!?;:\'-]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    df[Config.TICKET_SUMMARY] = df[Config.TICKET_SUMMARY].apply(clean_text)
    df[Config.INTERACTION_CONTENT] = df[Config.INTERACTION_CONTENT].apply(clean_text)
    return df

def translate_to_en(texts):
    return texts

def get_data_summary(df):
    print(f"Total records: {len(df)}")
    for col in ['Type 2', 'Type 3', 'Type 4']:
        if col in df.columns:
            print(f"{col}: {df[col].nunique()} classes")
