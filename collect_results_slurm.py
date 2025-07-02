import pandas as pd
from glob import glob

frompath = 'RSRM/202506*'
topath = '"RSRM/output/multirun_28_all_vars_1.csv"'

all_dfs = [pd.read_csv(csv) for csv in glob(frompath)]
final_df = pd.concat(all_dfs)
final_df.to_csv(topath)