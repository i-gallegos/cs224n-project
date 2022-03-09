import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dir = '../results/'

def plot_bar_graph(df, ax, title):
    df = df.pivot_table(index=['dataset'], columns=['experiment'])
    df.columns = df.columns.droplevel()
    df = df[['within-dataset', 'pre-simplified', 'post-simplified']]
    df = df.reindex(['tldr', 'tosdr', 'small_billsum'])
    df.plot.bar(rot=0, ax=ax, legend=False, xlabel='')
    ax.set_title(title)
    ax.set_ylim([0,0.5])
    return ax

# Load data
rouge_scores = pd.DataFrame(columns=['dataset', 'experiment', 'R-1', 'R-2', 'R-L'])
for dataset in ['tldr', 'tosdr', 'small_billsum']:
    within_dataset = pd.read_csv(dir+'within_dataset/rouge_'+dataset+'_test.csv')
    within_dataset['dataset'] = dataset
    within_dataset['experiment'] = 'within-dataset'

    pre_simplified = pd.read_csv(dir+'pre_simplified/rouge_'+dataset+'_test.csv')
    pre_simplified['dataset'] = dataset
    pre_simplified['experiment'] = 'pre-simplified'

    post_simplified = pd.read_csv(dir+'post_simplified/rouge_'+dataset+'_test.csv')
    post_simplified['dataset'] = dataset
    post_simplified['experiment'] = 'post-simplified'

    rouge_scores = pd.concat((rouge_scores, within_dataset, pre_simplified, post_simplified))

# Create figures
fig, ((ax1), (ax2), (ax3)) = plt.subplots(1, 3, figsize=(10,5))
ax1 = plot_bar_graph(rouge_scores.drop(columns=['R-2', 'R-L']), ax1, 'R-1')
ax2 = plot_bar_graph(rouge_scores.drop(columns=['R-1', 'R-L']), ax2, 'R-2')
ax3 = plot_bar_graph(rouge_scores.drop(columns=['R-1', 'R-2']), ax3, 'R-L')

fig.supylabel('ROUGE F-1 Score')
fig.supxlabel('Dataset')
fig.legend(['within-dataset', 'pre-simplified', 'post-simplified'],
           loc='upper center',
           ncol=3,
           bbox_to_anchor=(0.5, 0.94))

fig.suptitle('Simplification as Pre- or Post-Processing')
fig.subplots_adjust(top=0.81)
plt.savefig('simplification_effects.png')
