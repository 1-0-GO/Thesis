import argparse
import logging
import os
import sys
import time
import datetime
from typing import List, Tuple, Dict
import numpy as np
from pandas import read_csv
import pandas as pd
import json

# Add the current file's directory to Python's module search path, 
# allowing imports of local modules from this script's location.
# Set up paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
sys.path.append(SCRIPT_DIR)
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

from model.config import Config
from model.pipeline import Pipeline

parser = argparse.ArgumentParser(description='RSRM')

parser.add_argument(
    '--task',
    default='nguyen/1',
    type=str, help="""task place""")

parser.add_argument(
    '--num_test',
    default=1,
    type=int, help='number of tests performed, default 10')

parser.add_argument(
    '--json_path',
    default="config/config.json",
    type=str, help='configuration file path')

parser.add_argument(
    "--output",
    default="",
    type=str, help='output path')

parser.add_argument(
    "--threshold",
    default=1e-10,
    type=float, help="threshold for evaluation")

def parse_list(value):
    return value.split(',')

parser.add_argument(
    '--fit', 
    default=[],
    type=parse_list, 
    help='Columns to fit (e.g., --fit A,H)')

parser.add_argument(
    '--split', 
    default=[],
    type=parse_list, help='Columns to split (e.g., --split Species,Archipelago)')

def identify_header(path, n=5, th=0.9):
    df1 = read_csv(path, header='infer', nrows=n)
    df2 = read_csv(path, header=None, nrows=n)
    sim = (df1.dtypes.values == df2.dtypes.values).mean()
    return 'infer' if sim < th else None

def load_dataset(path):
    root = os.path.join(DATA_DIR, f"{path}_")
    header = identify_header(root + "train.csv")
    df_train = read_csv(root + "train.csv", header=header)
    df_test = read_csv(root + "test.csv", header=header)
    return df_train, df_test

def split_dataset(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    fit: List[str],
    split: List[str],
) -> Tuple[
    Dict[Tuple[str, ...], np.ndarray],  # x_train
    Dict[Tuple[str, ...], np.ndarray],  # t_train 
    Dict[Tuple[str, ...], np.ndarray],  # x_test
    Dict[Tuple[str, ...], np.ndarray]   # t_test
]:
    # Ensure target column is included
    target_col = df_train[[c for c in df_train.columns if c not in split]].columns[-1]
    if target_col not in fit:
        fit = fit + [target_col]
    maxim = df_train[fit].iloc[:, :-1].max().max()
    
    def process_df(df: pd.DataFrame) -> Tuple[Dict[Tuple[str, ...], np.ndarray], 
                                            Dict[Tuple[str, ...], np.ndarray]]:
        """Process dataframe into split dictionaries"""
        x = {}
        t = {}
        if not split:
            data = df[fit].values.T
            key = ("ALL",)
            x[key] = np.asarray(data[:-1])
            t[key] = np.asarray(data[-1])
        else:
            for group, df_group in df.groupby(split):
                group_data = df_group[fit].values.T
                group_key = group if isinstance(group, tuple) else (group,)
                x[group_key] = np.asarray(group_data[:-1])  # Features
                t[group_key] = np.asarray(group_data[-1])    # Targets
        return x, t
    
    x_train, t_train = process_df(df_train)
    x_test, t_test = process_df(df_test)
    return (x_train, t_train, x_test, t_test), maxim.item()

# Merge overrides
def deep_update(d, u):
    for k, v in u.items():
        if isinstance(v, dict) and isinstance(d.get(k), dict):
            deep_update(d[k], v)
        else:
            d[k] = v
    return d

def main(task, num_test, json_path, output, threshold, fit, split, extra_args={}):
    base_task = os.path.basename(task) + '_' + timestamp
    if output == "":
        output_dir = os.path.dirname(os.path.join(OUTPUT_DIR, task))
        os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = os.path.normpath(output)
    output = os.path.join(output_dir, base_task)

    config = Config()
    with open(json_path) as f:
        js = json.load(f)
        js = deep_update(js, extra_args)
    config.from_dict(js)
    config.output = output
    model = Pipeline(config=config)

    all_times = []
    all_eqs = []
    all_counts = []
    num_success = 0

    data = load_dataset(task)
    df_train, df_test = data
    all_fitting_cols = df_train.columns[:-1]
    if fit == []:
        fit = [c for c in all_fitting_cols if c not in split]
    data, maxim = split_dataset(*data, fit, split)

    for i_test in range(num_test):
        config.counter = [0, 0, 0]
        config.maxim = maxim
        sys.stdout.flush()
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        start_time = time.time()
        result = model.fit(*data)
        print(f"\rtask:{task} expr:{result[0]} Loss_{config.loss_name}:{result[1]:.2f} Test {i_test}/{num_test}.", end="")
        if threshold > result[1]:
            num_success += 1
        all_times.append(time.time() - start_time)
        all_eqs.append(result[0])
        all_counts.append(config.counter)
        

    with open(output + '_FINAL.txt', 'w') as output_file:
        for eq in all_eqs:
            if eq is not None:
                output_file.write(eq + '\n')

    print()
    print('final result:')
    print('success rate :', "{:.0%}".format(num_success / num_test))
    print('average discovery time is', np.round(np.mean(all_times), 3), 'seconds')
    print('Number of equations looked at (per test) [Total, Timed out, Successful]: ', all_counts)
    df = model.pf.to_df()
    df.to_csv(output + '.csv', index=False)
    return df

if __name__ == '__main__':
    args = parser.parse_args()
    main(args.task, args.num_test, args.json_path, args.output, args.threshold, args.fit, args.split)
