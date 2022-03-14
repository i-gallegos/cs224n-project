import pandas as pd

for dataset in ['small_billsum']:#, 'small_billsum']:
    print(dataset)
    df = pd.read_csv(dataset+'/'+dataset+'.csv')
    df['len'] = df['summary'].str.split().apply(len)
    df = df.sort_values(by=['len'])
    shortest = df.head(3).reset_index()

    for index, row in shortest.iterrows():
        doc = ' '.join(row["document"].split()[:128])+'...'
        print(f'{doc} & {row["summary"]} \\\\ \\midrule')

    print()
    print()
