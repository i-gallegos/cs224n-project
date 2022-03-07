import os

experiment = 'tiny_billsum'

dataset = 'tiny_billsum'
name = 'tiny_billsum_test'
checkpoint = 'tiny_billsum_lr3e-05_seed224_full_datasetFalse/checkpoint-52'
os.system(f"python evaluate_bart.py bart-large-cnn-finetuned/{checkpoint}/ \
            --test_file data/{dataset}/{dataset}_test.csv \
            --out_file results/{experiment}/preds_{name}.txt \
            --ref_file data/reference/reference_{dataset}_test.txt \
            --result_file results/{experiment}/rouge_{name}.csv")
