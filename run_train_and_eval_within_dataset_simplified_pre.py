import os

experiment = 'pre_simplified'

os.system(f"python fine_tune_bart.py tldr 0.00003 161 \
            --batch_size 16 \
            --grad_accumulation_steps 8")

dataset = 'tldr'
name = 'tldr_test'
checkpoint = 'tldr_lr2e-05_seed224_full_datasetFalse/checkpoint-4'
os.system(f"python evaluate_bart.py bart-large-cnn-finetuned/{checkpoint}/ \
            --test_file data/{dataset}/{dataset}_test_simplified.csv \
            --out_file results/{experiment}/preds_{name}.txt \
            --ref_file data/reference/reference_{dataset}_test.txt \
            --result_file results/{experiment}/rouge_{name}.csv")



os.system(f"python fine_tune_bart.py tosdr 0.00003 161 \
            --batch_size 16 \
            --grad_accumulation_steps 8")

dataset = 'tosdr'
name = 'tosdr_test'
checkpoint = 'tosdr_lr3e-05_seed161_full_datasetFalse/checkpoint-8'
os.system(f"python evaluate_bart.py bart-large-cnn-finetuned/{checkpoint}/ \
            --test_file data/{dataset}/{dataset}_test_simplified.csv \
            --out_file results/{experiment}/preds_{name}.txt \
            --ref_file data/reference/reference_{dataset}_test.txt \
            --result_file results/{experiment}/rouge_{name}.csv")



os.system(f"python fine_tune_bart.py small_billsum 0.00003 161 \
            --batch_size 16 \
            --grad_accumulation_steps 8")

dataset = 'small_billsum'
name = 'small_billsum_test'
checkpoint = 'small_billsum_lr3e-05_seed161_full_datasetFalse/checkpoint-44'
os.system(f"python evaluate_bart.py bart-large-cnn-finetuned/{checkpoint}/ \
            --test_file data/{dataset}/{dataset}_test_simplified.csv \
            --out_file results/{experiment}/preds_{name}.txt \
            --ref_file data/reference/reference_{dataset}_test.txt \
            --result_file results/{experiment}/rouge_{name}.csv")
