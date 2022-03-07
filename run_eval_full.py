import os

experiment = 'full_dataset'
"""
datasets = ['tosdr', 'small_billsum']
name = 'tldr_full'
checkpoint = 'tldr_lr3e-05_seed161_full_datasetTrue/checkpoint-4'
for dataset in datasets:
    os.system(f"python evaluate_bart.py bart-large-cnn-finetuned/{checkpoint}/ \
            --test_file data/{dataset}/{dataset}_test.csv \
            --out_file results/{experiment}/preds_{name}_on_{dataset}.txt \
            --ref_file data/reference/reference_{dataset}_test.txt \
            --result_file results/{experiment}/rouge_{name}_on_{dataset}.csv")
"""

#datasets = ['tldr', 'small_billsum']
datasets = ['small_billsum']
name = 'tosdr_full'
checkpoint = 'tosdr_lr3e-05_seed161_full_datasetTrue/checkpoint-8'
for dataset in datasets:
    os.system(f"python evaluate_bart.py bart-large-cnn-finetuned/{checkpoint}/ \
            --test_file data/{dataset}/{dataset}_test.csv \
            --out_file results/{experiment}/preds_{name}_on_{dataset}.txt \
            --ref_file data/reference/reference_{dataset}_test.txt \
            --result_file results/{experiment}/rouge_{name}_on_{dataset}.csv")

datasets = ['tosdr', 'tldr']
name = 'small_billsum_full'
checkpoint = 'small_billsum_lr3e-05_seed224_full_datasetTrue/checkpoint-52'
for dataset in datasets:
    os.system(f"python evaluate_bart.py bart-large-cnn-finetuned/{checkpoint}/ \
            --test_file data/{dataset}/{dataset}_test.csv \
            --out_file results/{experiment}/preds_{name}_on_{dataset}.txt \
            --ref_file data/reference/reference_{dataset}_test.txt \
            --result_file results/{experiment}/rouge_{name}_on_{dataset}.csv")



