import os
import pandas as pd
import numpy as np

for dataset in ['tldr', 'tosdr', 'small_billsum']:
    df = pd.read_csv(os.path.join(dataset, dataset+'.csv'))
    count = len(df)

    doc_len = np.asarray(df['document'].str.split().apply(len).tolist())
    summary_len = np.asarray(df['summary'].str.split().apply(len).tolist())

    mean_doc_len, mean_summary_len = np.mean(doc_len), np.mean(summary_len)
    std_doc_len, std_summary_len = np.std(doc_len), np.std(summary_len)

    print(f'DATASET: {dataset}')
    print(f'-- Dataset size (num total examples in train, dev, and test): {count}')
    print(f'-- Document length (num words): mean={mean_doc_len:.2f}, std={std_doc_len:.2f}')
    print(f'-- Summary length (num words): mean={mean_summary_len:.2f}, std={std_summary_len:.2f}')
    print()
