import os
import pandas as pd
from sklearn.model_selection import train_test_split

load_dir = '../billsum'

train = pd.read_csv(os.path.join(load_dir, 'billsum_train.csv'))
dev = pd.read_csv(os.path.join(load_dir, 'billsum_train.csv'))
test = pd.read_csv(os.path.join(load_dir, 'billsum_train.csv'))

df = pd.concat((train, dev, test))
small_df = df.sample(frac=0.25, replace=False, random_state=0)
small_df.to_csv('small_billsum.csv', index=False)

#70/15/15 split
train, test = train_test_split(small_df, test_size=0.3, random_state=0)
dev, test = train_test_split(test, test_size=0.5, random_state=0)

train.to_csv('small_billsum_train.csv', index=False)
dev.to_csv('small_billsum_dev.csv', index=False)
test.to_csv('small_billsum_test.csv', index=False)
