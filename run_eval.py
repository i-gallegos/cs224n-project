import os
<<<<<<< HEAD

experiment = 'full_dataset'

datasets = ['tosdr', 'small_billsum']
name = 'tldr_full'
checkpoint = 'tldr_batchsize16_lr3e-05_seed161_full_datasetTrue/checkpoint-4'
for dataset in datasets:
    os.system(f"python evaluate_bart.py bart-large-cnn-finetuned/{checkpoint}/ \
            --test_file data/{dataset}/{dataset}.csv \
            --out_file results/{experiment}/preds_{name}.txt \
            --ref_file data/reference/reference_{dataset}_full.txt \
            --result_file results/{experiment}/rouge_{name}_on_{dataset}.csv")

datasets = ['tldr', 'small_billsum']
name = 'tosdr_full'
checkpoint = 'tosdr_batchsize16_lr3e-05_seed161_full_datasetTrue/checkpoint-8'
for dataset in datasets:
    os.system(f"python evaluate_bart.py bart-large-cnn-finetuned/{checkpoint}/ \
            --test_file data/{dataset}/{dataset}.csv \
            --out_file results/{experiment}/preds_{name}.txt \
            --ref_file data/reference/reference_{dataset}_full.txt \
            --result_file results/{experiment}/rouge_{name}_on_{dataset}.csv")

datasets = ['tosdr', 'tldr']
name = 'small_billsum_full'
checkpoint = 'small_billsum_batchsize16_lr3e-05_seed161_full_datasetTrue/checkpoint-120'
for dataset in datasets:
    os.system(f"python evaluate_bart.py bart-large-cnn-finetuned/{checkpoint}/ \
            --test_file data/{dataset}/{dataset}.csv \
            --out_file results/{experiment}/preds_{name}.txt \
            --ref_file data/reference/reference_{dataset}_full.txt \
            --result_file results/{experiment}/rouge_{name}_on_{dataset}.csv")


=======
'''EXAMPLE
dataset = tldr
name = tldr_test
experiment = TODO
checkpoint = checkpoint-1

os.system(f"python evaluate_bart.py bart-large-cnn-finetuned/{checkpoint}/ \
            --test_file data/{dataset}/{name}.csv \ #{dataset}_test.csv for Isabel, {dataset}.csv for Kaylee
            --out_file results/{experiment}/preds_{name}.txt \ # file to save to
            --ref_file data/reference/reference_{name}.txt \ #reference_{dataset}_test.txt for Isabel, reference_{dataset}_full.txt for Kaylee
            --result_file results/{experiment}/rouge_{name}.csv") # file to save to
'''
>>>>>>> 3e4a72b1d66320b44739b2d821c9050d7fa36755
