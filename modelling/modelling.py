from model.randomforest import RandomForest
from model.linearsvm import LinearSVM
from Common.Data_Model import Data, HierarchicalDataContainer
from Config import Config
from utils import remove_nan_rows, remove_low_frequency_classes
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


MODEL_BUILDERS = [
    ("RandomForest", RandomForest),
    ("LinearSVM", LinearSVM),
]

RESULT_FILE_MAP = {
    "RandomForest": Path(__file__).resolve().parent.parent / "randomforest.csv",
    "LinearSVM": Path(__file__).resolve().parent.parent / "linar_svm.csv",
}

_written_result_files = set()


def save_results_to_csv(model, y_test, y_pred):
    model_class = model.__class__.__name__
    result_file = RESULT_FILE_MAP.get(model_class)
    if result_file is None:
        return

    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average='weighted', zero_division=0
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average='macro', zero_division=0
    )

    row = {
        'model_name': model.model_name,
        'accuracy': accuracy_score(y_test, y_pred),
        'weighted_precision': weighted_precision,
        'weighted_recall': weighted_recall,
        'weighted_f1': weighted_f1,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'sample_count': len(y_test),
    }

    write_mode = 'a'
    write_header = False
    if result_file not in _written_result_files:
        write_mode = 'w'
        write_header = True
        _written_result_files.add(result_file)

    pd.DataFrame([row]).to_csv(result_file, mode=write_mode, header=write_header, index=False)


def model_predict(data, df, name):
    for model_label, model_cls in MODEL_BUILDERS:
        model_name = f"{model_label} - {name}"
        model = model_cls(model_name=model_name, embeddings=data.get_embeddings(), y=data.get_type())
        model.train(data)
        y_pred = model.predict(data.get_X_test())
        model_evaluate(model, data.y_test, y_pred)

def model_evaluate(model, y_test, y_pred):
    model.print_results(y_test,y_pred)
    save_results_to_csv(model, y_test, y_pred)

def chained_multi_output(X, df):
    print("DESIGN CHOICE 1: CHAINED MULTI-OUTPUT CLASSIFICATION")
    type_cols = Config.TYPE_COLS
    separator = Config.CHAIN_SEPARATOR
    for level in range(1, len(type_cols) + 1):
        cols_to_chain = type_cols[:level]
        chain_name = separator.join(cols_to_chain)
        print(f"\nLevel {level}: Classifying chained label -> {chain_name}")
        df_work = df.copy()
        df_work['_orig_idx'] = np.arange(len(df_work))
        for col in cols_to_chain:
            df_work = remove_nan_rows(df_work, col)
        chained_label = df_work[cols_to_chain[0]].astype(str)
        for col in cols_to_chain[1:]:
            chained_label = chained_label + separator + df_work[col].astype(str)
        df_work['chained_target'] = chained_label
        df_work = remove_low_frequency_classes(df_work, 'chained_target')
        X_work = X[df_work['_orig_idx'].values]
        df_work = df_work.drop(columns=['_orig_idx']).reset_index(drop=True)
        if len(df_work) < 5:
            print(f"Skipping level {level}: not enough data after filtering.")
            continue
        data = Data(X_work, df_work, target_col='chained_target')
        model_name = f"Chained Level {level} ({chain_name})"
        model_predict(data, df_work, model_name)

def hierarchical_modelling(X, df):
    print("DESIGN CHOICE 2: HIERARCHICAL MODELLING")
    type_cols = Config.TYPE_COLS
    print(f"\nLevel 1: Classifying {type_cols[0]}")
    df_level1 = df.copy()
    df_level1['_orig_idx'] = np.arange(len(df_level1))
    df_level1 = remove_nan_rows(df_level1, type_cols[0])
    df_level1 = remove_low_frequency_classes(df_level1, type_cols[0])
    X_level1 = X[df_level1['_orig_idx'].values]
    df_level1 = df_level1.drop(columns=['_orig_idx']).reset_index(drop=True)
    data_level1 = Data(X_level1, df_level1, target_col=type_cols[0])
    model_predict(data_level1, df_level1, type_cols[0])
    if len(type_cols) > 1:
        type2_classes = df_level1[type_cols[0]].unique()
        print(f"\nLevel 2: Classifying {type_cols[1]} for each class in {type_cols[0]}")
        for cls in sorted(type2_classes):
            print(f"\n Filter: {type_cols[0]} = '{cls}'")
            mask = df_level1[type_cols[0]] == cls
            df_filtered = df_level1[mask].copy()
            df_filtered['_l1_idx'] = np.arange(mask.sum())
            X_filtered_all = X_level1[mask.values]
            df_filtered = remove_nan_rows(df_filtered, type_cols[1])
            df_filtered = remove_low_frequency_classes(df_filtered, type_cols[1])
            X_filtered = X_filtered_all[df_filtered['_l1_idx'].values]
            df_filtered = df_filtered.drop(columns=['_l1_idx']).reset_index(drop=True)
            if len(df_filtered) < 5 or df_filtered[type_cols[1]].nunique() < 2:
                print(f"Skipping: insufficient data for {type_cols[1]} under {type_cols[0]}='{cls}'")
                continue
            data_level2 = Data(X_filtered, df_filtered, target_col=type_cols[1])
            model_predict(data_level2, df_filtered, f"{type_cols[1]} | {type_cols[0]}='{cls}'")
            if len(type_cols) > 2:
                type3_classes = df_filtered[type_cols[1]].unique()
                for cls3 in sorted(type3_classes):
                    print(f"\nFilter: {type_cols[0]}='{cls}', {type_cols[1]}='{cls3}'")
                    mask3 = df_filtered[type_cols[1]] == cls3
                    df_filtered3 = df_filtered[mask3].copy()
                    df_filtered3['_l2_idx'] = np.arange(mask3.sum())
                    X_filtered3_all = X_filtered[mask3.values]
                    df_filtered3 = remove_nan_rows(df_filtered3, type_cols[2])
                    df_filtered3 = remove_low_frequency_classes(df_filtered3, type_cols[2])
                    X_filtered3 = X_filtered3_all[df_filtered3['_l2_idx'].values]
                    df_filtered3 = df_filtered3.drop(columns=['_l2_idx']).reset_index(drop=True)
                    if len(df_filtered3) < 5 or df_filtered3[type_cols[2]].nunique() < 2:
                        print(f"Skipping: insufficient data for {type_cols[2]} under {type_cols[0]}='{cls}', {type_cols[1]}='{cls3}'")
                        continue
                    data_level3 = Data(X_filtered3, df_filtered3, target_col=type_cols[2])
                    model_predict(data_level3, df_filtered3, f"{type_cols[2]} | {type_cols[0]}='{cls}', {type_cols[1]}='{cls3}'")
