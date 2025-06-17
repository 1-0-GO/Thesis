import pandas as pd
import numpy as np
from eggp import EGGP
from pandas import read_csv
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
feature_cols = ['A', 'T']
group_cols = ['Archipelago', 'species']
target_col = 'y'
df = read_csv(os.path.join(SCRIPT_DIR, 'RSRM/data/GDM_train.csv'))
df = df.loc[df['species'] != 'Spiderd (endemics)']

# Prepare grouped views
X_views = []
y_views = []

for _, group in df.groupby(group_cols):
    X = group[feature_cols].to_numpy()
    y = group[target_col].to_numpy()
    
    # Skip empty or too-small groups
    if len(X) < 5:
        continue
    
    X_views.append(X)
    y_views.append(y)

# Check how many views were created
print(f"Prepared {len(X_views)} views.")
# Run multi-view symbolic regression
model = EGGP(gen=2000, nPop=200, nonterminals="add,sub,mul,div,exp,log,tanh,square,sqrt", loss="Poisson")
model.fit_mvsr(X_views, y_views)

model.results.to_csv('a.csv')

# Predict for each view
# predictions = [model.predict_mvsr(X_views[i], view=i) for i in range(len(X_views))]

# Example: Print predictions for the first view
# print(predictions[0])
