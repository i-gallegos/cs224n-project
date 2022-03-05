import os
import pandas as pd

simplified_dir = '../access/access/preds'

for dataset in ['small_billsum']:
    for split in ['test']: #['train', 'dev', 'test']:
        filepath = os.path.join(simplified_dir, dataset, 'preds_'+dataset+'_'+split)
        with open(filepath) as file:
            simplified = [line.rstrip() for line in file]
            print(simplified[0])

        df = pd.read_csv(os.path.join(dataset, dataset+'_'+split+'.csv'))
        df = df.drop(columns=['summary'])
        df['summary'] = simplified
        print(df['summary'])
        df.to_csv(os.path.join(dataset, dataset+'_'+split+'_simplified.csv'), index=False)
