import os
import pandas as pd

simplified_dir = '../access/access/preds'

for dataset in ['tldr', 'tosdr', 'small_billsum']:
    for split in ['test']: #['train', 'dev', 'test']:
        filepath = os.path.join(simplified_dir, dataset, 'preds_'+dataset+'_'+split)
        with open(filepath) as file:
            simplified = [line.rstrip() for line in file]

        df = pd.read_csv(os.path.join(dataset, dataset+'_'+split+'.csv'))
        df = df.drop(columns=['document'])
        df['document'] = simplified
        df.to_csv(os.path.join(dataset, dataset+'_'+split+'_simplified.csv'), index=False)
