import pandas as pd
import matplotlib.pyplot as plt
from itertools import cycle, islice

metrics = pd.read_csv('qualitative_metrics.csv')

#Prediction Quality,Prediction Match,Reference Quality


def stacked_bar(type, ax):
    df = metrics[[type,'Reference Quality']]
    df = df.groupby([type])

    names = {1:'Poor', 2:'Moderate', 3:'Good'}
    dist = pd.DataFrame(columns=['Poor', 'Moderate', 'Good'])
    for key, item in df:
        group = df.get_group(key).reset_index(drop=True)
        level = group.iloc[0][type]
        counts = group.groupby(['Reference Quality']).size()

        counts_df = pd.DataFrame.from_dict({'Poor':[counts[1]] if 1 in counts.index else 0,
                                            'Moderate':[counts[2]] if 2 in counts.index else 0,
                                            'Good':[counts[3]] if 3 in counts.index else 0,
                                            type:[names[level]]})
        counts_df = counts_df.set_index(type)
        dist = pd.concat((dist, counts_df))

    print(dist)
    stacked_data = dist.apply(lambda x: x/sum(x), axis=1)
    print(stacked_data)
    stacked_data.plot.bar(stacked=True, ax=ax, rot=1, xlabel=type, ylabel='Proportion', color=['r', '#FFD700', 'g'], legend=False)
    ax.set_title(type+' v. Reference Quality')


fig, ((ax1), (ax2)) = plt.subplots(1, 2, figsize=(10,5))
stacked_bar('Prediction Quality', ax1)
stacked_bar('Prediction Match', ax2)
fig.subplots_adjust(top=0.77)
fig.suptitle('Prediction v. Reference Qualities')
fig.legend(['Poor', 'Moderate', 'Good'], title='Reference Quality', loc='upper center', ncol=3, bbox_to_anchor=(0.5, 0.94))
plt.savefig('stacked_bar.png')
