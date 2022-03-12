import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_bar_graph(df, ax, title):
    df = df.pivot_table(index=['dataset'], columns=['method'])
    df.columns = df.columns.droplevel()
    # df = df[['across-dataset-tldr', 'across-dataset-tosdr', 'across-dataset-small-billsum', 'within-dataset']]
    # df = df.reindex(['tldr', 'tosdr', 'small_billsum'])

    df.plot.bar(rot=0, ax=ax, legend=True, xlabel='')
    ax.set_title(title)
    ax.set_ylim([0,0.5])
    return ax

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

print(results)
fig, ((ax1), (ax2), (ax3)) = plt.subplots(1, 3, figsize=(10,5))
ax1 = plot_bar_graph(results.drop(columns=['R-2', 'R-L']), ax1, 'R-1')
ax2 = plot_bar_graph(results.drop(columns=['R-1', 'R-L']), ax2, 'R-2')
ax3 = plot_bar_graph(results.drop(columns=['R-1', 'R-2']), ax3, 'R-L')
plt.savefig('baseline_within_dataset_compare.png')
