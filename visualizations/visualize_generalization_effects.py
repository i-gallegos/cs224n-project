import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dir = '../results/'

def plot_bar_graph(df, ax, title):
    df = df.pivot_table(index=['test_set'], columns=['model'])
    df.columns = df.columns.droplevel()
    df = df[['across-dataset-tldr', 'across-dataset-tosdr', 'across-dataset-small-billsum', 'within-dataset']]
    df = df.reindex(['tldr', 'tosdr', 'small_billsum'])

    df.plot.bar(rot=0, ax=ax, legend=False, xlabel='')
    ax.set_title(title)
    ax.set_ylim([0,0.5])
    return ax

# Load data
rouge_scores = pd.DataFrame(columns=['test_set', 'model', 'R-1', 'R-2', 'R-L'])
for model in ['tldr', 'tosdr', 'small_billsum']:

    for test_set in  ['tldr', 'tosdr', 'small_billsum']:
        if test_set == model:
            within_dataset = pd.read_csv(dir+'within_dataset/rouge_'+model+'_test.csv')
            within_dataset['test_set'] = test_set
            within_dataset['model'] = 'within-dataset'
            rouge_scores = pd.concat((rouge_scores, within_dataset))
        else:
            across_dataset = pd.read_csv(dir+'full_dataset/rouge_'+model+'_full_on_'+test_set+'.csv')
            across_dataset['test_set'] = test_set
            across_dataset['model'] = 'across-dataset-'+('-'.join(model.split('_')))
            rouge_scores = pd.concat((rouge_scores, across_dataset))


# Create figures
fig, ((ax1), (ax2), (ax3)) = plt.subplots(1, 3, figsize=(10,5))
ax1 = plot_bar_graph(rouge_scores.drop(columns=['R-2', 'R-L']), ax1, 'R-1')
ax2 = plot_bar_graph(rouge_scores.drop(columns=['R-1', 'R-L']), ax2, 'R-2')
ax3 = plot_bar_graph(rouge_scores.drop(columns=['R-1', 'R-2']), ax3, 'R-L')

fig.supylabel('ROUGE F-1 Score')
fig.supxlabel('Test Dataset')
fig.legend(['across-dataset-tldr', 'across-dataset-tosdr', 'across-dataset-small-billsum', 'within-dataset'],
           loc='upper center',
           ncol=4,
           bbox_to_anchor=(0.5, 0.94))
fig.suptitle('Generalization Across Datasets')
fig.subplots_adjust(top=0.81)
plt.savefig('generalization_effects.png')
