import os
import pandas as pd
from sklearn.model_selection import train_test_split

load_dir = '../small_billsum'

train_size = 60 # tldr train size
dev_size = 14   # tldr dev size

train = pd.read_csv(os.path.join(load_dir, 'small_billsum_train.csv'))
dev = pd.read_csv(os.path.join(load_dir, 'small_billsum_dev.csv'))
test = pd.read_csv(os.path.join(load_dir, 'small_billsum_test.csv'))

train = train.sample(n=train_size, random_state=0)
dev = dev.sample(n=dev_size, random_state=0)

train.to_csv('tiny_billsum_train.csv', index=False)
dev.to_csv('tiny_billsum_dev.csv', index=False)
test.to_csv('tiny_billsum_test.csv', index=False)
