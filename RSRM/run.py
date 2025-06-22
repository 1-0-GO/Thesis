from main import main, load_dataset, DATA_DIR, OUTPUT_DIR, timestamp
from pandas import concat, DataFrame
import hashlib
from os.path import join, isfile
from os import remove
from itertools import combinations
from glob import glob

# Parameters
task = 'GDM'
num_test = 1
json_path = 'ecology.json'
threshold = 1e-10
fit = ['A', 'T']
split = ['Archipelago', 'species']
output = ""

json_path = 'config/' + json_path
df, _ =  load_dataset(task)
df = df.loc[df['species'] != 'Spiderd (endemics)']
results_pf_df_list = []

def hash_dataframe(df):
    hash_object = hashlib.sha256(df.to_string().encode())
    hash_value = hash_object.hexdigest()

    return hash_value

def cleanup(temp_files):
    if isinstance(temp_files[0], list):
        temp_files = [path for sublist in temp_files for path in sublist]
    for file_path in temp_files:
        try:
            if isfile(file_path):
                remove(file_path)
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")

def run_df(df):
    # Create unique ID for this df
    hash_value = hash_dataframe(df)
    id_str = f"temp_{hash_value}"

    train_path = join(DATA_DIR, f"{id_str}_train.csv")
    test_path = join(DATA_DIR, f"{id_str}_test.csv")

    df.to_csv(train_path)
    df.to_csv(test_path)

    # Run main logic
    results_pf_df = main(id_str, num_test, json_path, output, threshold, fit, split)

    res_files = glob(join(OUTPUT_DIR, f'{id_str}*'))
    files_to_cleanup = [train_path, test_path]
    files_to_cleanup.extend(res_files)
    return results_pf_df, files_to_cleanup

def leave_out_n_groups(n=1):
    grouped = dict(tuple(df.groupby(split)))  # creates a dictionary {group_key: group_df}
    group_keys = list(grouped.keys())
    N = max(1, len(group_keys) - n)
    group_combinations = list(combinations(group_keys, N))
    results = []
    for combo in group_combinations:
        df_combo =  concat([grouped[key] for key in combo], ignore_index=True)
        res = run_df(df_combo)
        results.append(res)
    results_pf_df_list, temp_files = zip(*results)
    cleanup(temp_files)
    return results_pf_df_list

results_pf_df_list = leave_out_n_groups(n=1)

final_results_df = concat(results_pf_df_list, ignore_index=True)
final_results_df.to_csv(join(OUTPUT_DIR, f'multirun_{timestamp}.csv'))