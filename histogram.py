import pandas as pd
import matplotlib.pyplot as plt
import os

from parameters import *

dataset = pd.DataFrame(columns=['ClassID', 'Size'])

entries = os.listdir(TRAIN_PATH)
count = 0
for entry in entries:
    dataset.loc[count] = [int(entry), len(os.listdir(TRAIN_PATH + "/" + entry))]
    count += 1

dat = dataset.sort_values(by=['ClassID'])

dat.plot(x='ClassID', y='Size', figsize=(8, 6), kind='bar', legend=False)
plt.xticks(rotation=0)
plt.show()
