import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_bar_graph(df, ax, title):
    df = df.pivot_table(index=['baseline'], columns=['dataset'])
    df.plot.bar(rot=0, ax=ax, legend=False, xlabel='')
    ax.set_title(title)
    ax.set_ylim([0,0.3])
    return ax

# Load data
rouge_path = '../../results/baselines/baseline_rouge.csv'
rouge = pd.read_csv(rouge_path)
rouge_test = rouge[rouge['split'] == 'test'].drop(columns=['split'])
print(rouge_test)

# Create figures
fig, ((ax1), (ax2), (ax3)) = plt.subplots(3, 1, figsize=(10,9))
ax1 = plot_bar_graph(rouge_test.drop(columns=['R-2', 'R-L']), ax1, 'R-1')
ax2 = plot_bar_graph(rouge_test.drop(columns=['R-1', 'R-L']), ax2, 'R-2')
ax3 = plot_bar_graph(rouge_test.drop(columns=['R-1', 'R-2']), ax3, 'R-L')

fig.supylabel('ROUGE F-1 Score')
fig.supxlabel('Baseline Method')
fig.legend(['TLDR', 'TOSDR'])
fig.subplots_adjust(hspace=0.4)
plt.savefig('baseline_simplified_rouge.png')
