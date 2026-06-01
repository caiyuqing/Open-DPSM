#%%
import os

import pandas as pd
inputDir = r"d:\Users\7009291\Desktop\Open-DPSM\v4\Example\Input\Eyetracking\s2"
for file in os.listdir(inputDir):
    print(file)
    df = pd.read_csv(os.path.join(inputDir, file), header=0, index_col=0)
    df.to_csv(os.path.join(inputDir, file), index = False, header = False)
