import os
import pandas as pd


for dataset in ['tldr', 'tosdr', 'small_billsum']:
    train_path = os.path.join(dataset, dataset+'_train.csv')
    dev_path = os.path.join(dataset, dataset+'_dev.csv')

    df_train = pd.read_csv(train_path)
    df_dev = pd.read_csv(dev_path)
    df = pd.concat([df_train, df_dev])
    df.to_csv(os.path.join(dataset, dataset+'_train_and_dev.csv'), index=False)
