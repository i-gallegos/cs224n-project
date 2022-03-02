import os
datasets = ['tldr', 'tosdr', 'small_billsum']
lrs = ['0.00001', '0.00002', '0.00003']
seeds = ['224', '161']
for dataset in datasets:
    for lr in lrs:
        for seed in seeds:
            os.system(f"python fine_tune_bart.py {dataset} {lr} {seed} --batch_size 32 --grad_accumulation_steps 4 --train_on_full_dataset False")
