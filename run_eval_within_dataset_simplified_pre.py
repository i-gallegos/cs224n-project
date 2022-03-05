import os

experiment = 'pre_simplified'

dataset = 'tldr'
name = 'tldr_test'
checkpoint = 'tldr_lr2e-05_seed224_full_datasetFalse/checkpoint-4'
os.system(f"python evaluate_bart.py bart-large-cnn-finetuned/{checkpoint}/ \
            --test_file data/{dataset}/{dataset}_test_simplified.csv \
            --out_file results/{experiment}/preds_{name}.txt \
            --ref_file data/reference/reference_{dataset}_test.txt \
            --result_file results/{experiment}/rouge_{name}.csv")

dataset = 'tosdr'
name = 'tosdr_test'
checkpoint = 'tosdr_lr3e-05_seed161_full_datasetFalse/checkpoint-8'
os.system(f"python evaluate_bart.py bart-large-cnn-finetuned/{checkpoint}/ \
            --test_file data/{dataset}/{dataset}_test_simplified.csv \
            --out_file results/{experiment}/preds_{name}.txt \
            --ref_file data/reference/reference_{dataset}_test.txt \
            --result_file results/{experiment}/rouge_{name}.csv")

dataset = 'small_billsum'
name = 'small_billsum_test'
checkpoint = 'small_billsum_lr3e-05_seed224_full_datasetFalse/checkpoint-120'
os.system(f"python evaluate_bart.py bart-large-cnn-finetuned/{checkpoint}/ \
            --test_file data/{dataset}/{dataset}_test_simplified.csv \
            --out_file results/{experiment}/preds_{name}.txt \
            --ref_file data/reference/reference_{dataset}_test.txt \
            --result_file results/{experiment}/rouge_{name}.csv")
