import pandas as pd
import os

def read_csv(dataset):
    cur_path = os.getcwd() 
    dataset_path = f'{cur_path}/../data/{dataset}'
    phases = ['test', 'train', 'dev']
    with open(f'reference_{dataset}_full.txt', "w") as ref_file:
        for phase in phases:
            csv_file = f'{dataset_path}/{dataset}_{phase}.csv'
            with open(csv_file, 'r') as input_file:
                col_list = ['document', 'summary', 'id']
                df = pd.read_csv(input_file, usecols=col_list)
                
                for row in df['summary']:
                    ref_file.write("".join(row + '\n'))
    ref_file.close()
    return f'reference_{dataset}_full.txt'

read_csv('tldr')
read_csv('tosdr')
read_csv('small_billsum')