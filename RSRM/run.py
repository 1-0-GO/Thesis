from main import main, load_dataset, DATA_DIR, OUTPUT_DIR, timestamp
from pandas import concat
from os.path import join, isfile
from os import remove
from glob import glob

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
for group, df_group in df.groupby(split):
    df_comp = df[~df.index.isin(df_group.index)]
    df_comp.to_csv(join(DATA_DIR, 'temp_train.csv'))
    df_comp.to_csv(join(DATA_DIR, 'temp_test.csv'))
    results_pf_df = main('temp', num_test, json_path, output, threshold, fit, split)
    results_pf_df['left_out_archipelago'] = group[0]
    results_pf_df['left_out_species'] = group[1]
    results_pf_df_list.append(results_pf_df)
prefix = "temp*"  # Matches all files starting with "temp"

# Get a list of all matching files (including full paths)
matching_files = glob(join(OUTPUT_DIR, prefix))
matching_files.extend(glob(join(DATA_DIR, prefix)))

# Delete each file
for file_path in matching_files:
    try:
        if isfile(file_path):  # Ensure it's a file (not a directory)
            remove(file_path)
    except Exception as e:
        print(f"Error deleting {file_path}: {e}")
final_results_df = concat(results_pf_df_list)
final_results_df.to_csv(join(OUTPUT_DIR, f'multirun_{timestamp}.csv'))