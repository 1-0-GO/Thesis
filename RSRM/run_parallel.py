from main import main, load_dataset, DATA_DIR, OUTPUT_DIR, timestamp
from pandas import concat
from os.path import join, isfile
from os import remove
from glob import glob
import pandas as pd

# Constants
task = 'GDM'
num_test = 1
json_path = 'config/ecology.json'
threshold = 1e-10
fit = ['A', 'T']
split = ['Archipelago', 'species']
output = ""

# --- Define the parallel function at the top level ---
def process_group(group_tuple):
    from main import main, load_dataset, DATA_DIR, OUTPUT_DIR  # Ensure paths/constants exist in subprocess

    group, df_group = group_tuple
    df, _ = load_dataset('GDM')
    df = df.loc[df['species'] != 'Spiderd (endemics)']
    df_comp = df[~df.index.isin(df_group.index)]

    # Create unique ID for this group
    id_str = f"temp_{group[0]}_{group[1]}".replace(" ", "_").replace("/", "_")

    train_path = join(DATA_DIR, f"{id_str}_train.csv")
    test_path = join(DATA_DIR, f"{id_str}_test.csv")

    df_comp.to_csv(train_path)
    df_comp.to_csv(test_path)

    # Run main logic
    results_pf_df = main(id_str, num_test, json_path, output, threshold, fit, split)

    # Add metadata
    results_pf_df['left_out_archipelago'] = group[0]
    results_pf_df['left_out_species'] = group[1]

    return results_pf_df, [train_path, test_path]

# --- Main ---
if __name__ == '__main__':
    from concurrent.futures import ProcessPoolExecutor

    # Load and filter the dataset
    df, _ = load_dataset(task)
    df = df.loc[df['species'] != 'Spiderd (endemics)']

    # Group by the split variables
    grouped = list(df.groupby(split))

    # Run all groups in parallel
    with ProcessPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(process_group, grouped))

    # Unpack results
    results_pf_df_list, temp_files = zip(*results)
    results_pf_df_list = list(results_pf_df_list)
    flat_temp_files = [path for sublist in temp_files for path in sublist]

    # Save final combined results
    final_results_df = concat(results_pf_df_list)
    final_results_df.to_csv(join(OUTPUT_DIR, f'multirun_{timestamp}.csv'), index=False)

    # Cleanup temp files
    prefix = "temp*"
    matching_files = glob(join(OUTPUT_DIR, prefix))
    matching_files.extend(flat_temp_files)

    for file_path in matching_files:
        try:
            if isfile(file_path):
                remove(file_path)
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")
