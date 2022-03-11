import os
import pandas as pd
import numpy as np

info = pd.DataFrame(columns=['Dataset', 'Total Examples', 'Mean Document Length (Std)', 'Mean Summary Length (Std)'])
for dataset in ['tldr', 'tosdr', 'small_billsum', 'tiny_billsum']:
    df_train = pd.read_csv(os.path.join('../data', dataset, dataset+'_train.csv'))
    df_dev = pd.read_csv(os.path.join('../data', dataset, dataset+'_dev.csv'))
    df_test = pd.read_csv(os.path.join('../data', dataset, dataset+'_test.csv'))

    if dataset == 'tiny_billsum':
        df = pd.concat((df_train, df_dev, df_test))
    else:
        df = pd.read_csv(os.path.join('../data', dataset, dataset+'.csv'))

    train = len(df_train)
    dev = len(df_dev)
    test = len(df_test)
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

    new_df = pd.DataFrame.from_dict({'Dataset': [dataset],
                                     'Total Examples': f'{train}/{dev}/{test} ({count})',
                                     'Mean Document Length (Std)': f'{mean_doc_len:.2f} ({std_doc_len:.2f})',
                                     'Mean Summary Length (Std)': f'{mean_summary_len:.2f} ({std_summary_len:.2f})'})
    info = pd.concat((info, new_df))

print(info)
info.to_latex('dataset_characteristics_latex.txt', index=False)
