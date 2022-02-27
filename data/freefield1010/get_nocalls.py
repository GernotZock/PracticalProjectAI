import os
import pandas as pd

df = pd.read_csv("metadata.csv")
for i, (item_id, hasbird) in df.iterrows():
    path = os.path.join("wav", str(item_id) + ".wav")
    if os.path.exists(path):
        # remove files that have birds in them
        if hasbird == 1:
            os.remove(path)
