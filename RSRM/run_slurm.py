from main import main, load_dataset, DATA_DIR, OUTPUT_DIR, timestamp
from pandas import concat
import hashlib
from os.path import join, isfile
from os import remove
from itertools import combinations
from glob import glob
import sys

# Parameters
task = 'GDM'
num_test = 1
json_path = 'config/ecology.json'
threshold = 1e-10
fit = ['A', 'T', 'H', 'Dm', 'Di', 'Do']
split = ['Archipelago', 'species']
output = ""
n_groups_out = 2

# Get SLURM_ARRAY_TASK_ID from command line or environment
try:
    idx = int(sys.argv[1])
except IndexError:
    raise ValueError("Please provide the SLURM_ARRAY_TASK_ID as the first argument.")

# Load and filter the dataset
df, _ = load_dataset(task)
df = df.loc[df['species'] != 'Spiderd (endemics)']

# Grouping and preparing combinations
grouped = dict(tuple(df.groupby(split)))
group_keys = list(grouped.keys())
N = max(1, len(group_keys) - n_groups_out)
group_combinations = list(combinations(group_keys, N))

if idx >= len(group_combinations):
    raise ValueError(f"Invalid SLURM_ARRAY_TASK_ID {idx}: exceeds number of combinations ({len(group_combinations)})")

def hash_dataframe(df):
    hash_object = hashlib.sha256(df.to_string().encode())
    return hash_object.hexdigest()

def cleanup(temp_files):
    for file_path in temp_files:
        try:
            if isfile(file_path):
                remove(file_path)
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")

def run_df(df):
    hash_value = hash_dataframe(df)
    id_str = f"temp_{hash_value}"

    train_path = join(DATA_DIR, f"{id_str}_train.csv")
    test_path = join(DATA_DIR, f"{id_str}_test.csv")

    df.to_csv(train_path)
    df.to_csv(test_path)

    results_pf_df = main(id_str, num_test, json_path, output, threshold, fit, split)

    res_files = glob(join(OUTPUT_DIR, f'{id_str}*'))
    files_to_cleanup = [train_path, test_path] + res_files
    cleanup(files_to_cleanup)

    return results_pf_df

# Run for the indexed group combo
df_combo = concat([grouped[key] for key in group_combinations[idx]], ignore_index=True)
result_df = run_df(df_combo)

# Save individual result (you can merge all after array ends)
result_df.to_csv(join(OUTPUT_DIR, f'results_{timestamp}_{idx}.csv'), index=False)
