import os
import pandas as pd

for dataset in ['tldr', 'tosdr', 'small_billsum']:
    i = 0
    for split in ['train', 'dev', 'test']:
        path = os.path.join(dataset, dataset+'_'+split+'.csv')
        df = pd.read_csv(path)
        df['id'] = range(i, i+len(df))
        df.to_csv(path, index=False)
        i += len(df)
