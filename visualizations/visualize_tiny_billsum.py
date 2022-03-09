import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dir = '../results/'

def plot_bar_graph(df, ax, title):
    df = df.pivot_table(columns=['dataset'])
    print(df)
    df = df[['within-dataset-tldr', 'within-dataset-tiny-billsum', 'within-dataset-billsum']]
    df.plot.bar(rot=0, ax=ax, legend=False, xlabel='')
    ax.set_title(title)
    ax.set_ylim([0,0.5])
    ax.set_xticks([])
    return ax

# Load data
rouge_scores = pd.DataFrame(columns=['dataset', 'R-1', 'R-2', 'R-L'])
for dataset in ['tldr', 'tiny_billsum', 'small_billsum']:
    df = pd.read_csv(dir+'within_dataset/rouge_'+dataset+'_test.csv')
    if dataset == 'small_billsum':
        dataset = 'billsum'
    df['dataset'] = 'within-dataset-'+('-'.join(dataset.split('_')))
    rouge_scores = pd.concat((rouge_scores, df))


# Create figures
fig, ((ax1), (ax2), (ax3)) = plt.subplots(1, 3, figsize=(10,5))
ax1 = plot_bar_graph(rouge_scores.drop(columns=['R-2', 'R-L']), ax1, 'R-1')
ax2 = plot_bar_graph(rouge_scores.drop(columns=['R-1', 'R-L']), ax2, 'R-2')
ax3 = plot_bar_graph(rouge_scores.drop(columns=['R-1', 'R-2']), ax3, 'R-L')

fig.supylabel('ROUGE F-1 Score')
fig.legend(['within-dataset-tldr (59)', 'within-dataset-tiny-billsum (59)', 'within-dataset-billsum (1412)'],
           title='Model (Training Size)',
           loc='upper center',
           ncol=3,
           bbox_to_anchor=(0.5, 0.94))

fig.suptitle('Performance v. Training Set Size')
fig.subplots_adjust(top=0.75)
plt.savefig('tiny_billsum.png')
