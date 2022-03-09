import os
import pandas as pd
import numpy as np

baseline_results = pd.read_csv('../results/baselines/baseline_rouge.csv').drop(columns=['split'])
baseline_results = baseline_results.rename(columns={'baseline': 'method'})

fine_tuned_results = pd.DataFrame(columns=['dataset', 'method', 'R-1', 'R-2', 'R-L'])
for dataset in ['tldr', 'tosdr', 'small_billsum']:
    within_dataset = pd.read_csv('../results/within_dataset/rouge_'+dataset+'_test.csv')
    within_dataset['dataset'] = dataset
    within_dataset['method'] = 'fine_tuned_bart_large_cnn'

    fine_tuned_results = pd.concat((fine_tuned_results, within_dataset))

results = pd.concat((baseline_results, fine_tuned_results))
results = results.replace('tldr', 'TLDR')
results = results.replace('tosdr', 'TOSDR')
results = results.replace('small_billsum', 'Billsum')
results = results.replace('text_rank', 'TextRank')
results = results.replace('kl_sum', 'KLSum')
results = results.replace('lead_one', 'Lead-1')
results = results.replace('lead_k', 'Lead-K')
results = results.replace('random_k', 'Random-K')
results = results.replace('bart', 'bart-large-cnn')
results = results.replace('fine_tuned_bart_large_cnn', 'Fine-Tuned bart-large-cnn')

results = results.pivot_table(index=['method'], columns=['dataset'])
results = results.reindex(['TextRank', 'KLSum', 'Lead-1', 'Lead-K', 'Random-K', 'bart-large-cnn', 'Fine-Tuned bart-large-cnn'])
results = results[[('R-1', 'TLDR'),
                   ('R-1', 'TOSDR'),
                   ('R-1', 'Billsum'),
                   ('R-2', 'TLDR'),
                   ('R-2', 'TOSDR'),
                   ('R-2', 'Billsum'),
                   ('R-L', 'TLDR'),
                   ('R-L', 'TOSDR'),
                   ('R-L', 'Billsum')]]

cols = results.select_dtypes(np.number).columns
results[cols] = results[cols].mul(100)
results = np.round(results, decimals=2)
results[cols] = results[cols].astype(str)
print(results)

s = results.style.highlight_max(props='textbf:--rwrap;')
s.to_latex('baseline_within_dataset_compare_latex.txt', multicol_align='c')
