# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os
from itertools import product
from pathlib import Path
import pandas as pd
import csv

REPO_DIR = Path(__file__).resolve().parent.parent.parent
EXP_DIR = REPO_DIR / 'experiments'
RESOURCES_DIR = REPO_DIR / 'resources'
DATASETS_DIR = RESOURCES_DIR / 'datasets'
DATA_tldr = 'data/tldr'
DATA_tosdr = 'data/tosdr'
VARIOUS_DIR = RESOURCES_DIR / 'various'
FASTTEXT_EMBEDDINGS_PATH = VARIOUS_DIR / 'fasttext-vectors/wiki.en.vec'
MODELS_DIR = RESOURCES_DIR / 'models'
BEST_MODEL_DIR = MODELS_DIR / 'best_model'

LANGUAGES = ['complex', 'simple']
PHASES = ['train', 'valid', 'test']

def get_dataset_dir(dataset):
    return DATASETS_DIR / dataset


def get_data_filepath(dataset, phase, language, i=None):
    suffix = ''  # Create suffix e.g. for multiple references
    if i is not None:
        suffix = f'.{i}'
    filename = f'{dataset}.{phase}.{language}{suffix}'
    return get_dataset_dir(dataset) / filename


def get_filepaths_dict(dataset):
    return {(phase, language): get_data_filepath(dataset, phase, language)
            for phase, language in product(PHASES, LANGUAGES)}

def read_csv(csv_file, dataset, phase):
    with open(csv_file, 'r') as input_file:
        col_list = ['document', 'summary']
        df = pd.read_csv(input_file, usecols=col_list)

    # create files with data from csv
    with open(f'original_{dataset}_{phase}.txt', "w") as orig_file:
        for row in df['document']:
            row = row.replace("\n", "")
            orig_file.write("".join(row)+'\n')
        orig_file.close()

    with open(f'reference_{dataset}_{phase}.txt', "w") as ref_file:
        for row in df['summary']:
            row = row.replace("\n", "")
            ref_file.write("".join(row)+'\n')
        ref_file.close()
    return f'original_{dataset}_{phase}.txt', f'reference_{dataset}_{phase}.txt'

# be in cs224n-project/access
def get_law_filepath(dataset, phase):
    cur_path = os.getcwd() 
    dataset_path = f'{cur_path}/../data/{dataset}'
    filename = f'{dataset_path}/{dataset}_{phase}.csv'
    return read_csv(filename, dataset, phase)

# edited for post summary
def get_pred_filepath(dataset, phase):
    cur_path = os.getcwd() 
    pred_file = open(f'{cur_path}/access/preds/{dataset}/preds_{dataset}_{phase}', 'w')
    pred_file.close()
    pred_filepath = f'{cur_path}/access/preds/{dataset}/preds_{dataset}_{phase}'
    return pred_filepath

def get_pred_filepath_post(dataset, phase, sum_model):
    cur_path = os.getcwd() 
    pred_file = open(f'{cur_path}/access/preds/preds_{dataset}_{phase}_{sum_model}', 'w')
    pred_file.close()
    pred_filepath = f'{cur_path}/access/preds/preds_{dataset}_{phase}_{sum_model}'
    return pred_filepath

def get_post_filepath(dataset, phase, sum_model):
    cur_path = os.getcwd() 
    dataset_path = f'{cur_path}/../results/baselines/{dataset}/{phase}'
    return f'{dataset_path}/{sum_model}.txt', f'{dataset_path}/ref.txt'

# 
def get_pre_summarization(dataset):
    cur_path = os.getcwd()
    dataset_path = f'{cur_path}/../results/within_dataset/preds_{dataset}_test.txt'
    reference_path = f'{cur_path}/../data/reference/reference_{dataset}_test.txt'
    return dataset_path, reference_path