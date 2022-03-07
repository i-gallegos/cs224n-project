import os

experiment = 'full_dataset'

datasets = ['tosdr', 'small_billsum']
name = 'tldr_full'
checkpoint = 'tldr_batchsize16_lr3e-05_seed161_full_datasetTrue/checkpoint-4'
for dataset in datasets:
    os.system(f"python evaluate_bart.py bart-large-cnn-finetuned/{checkpoint}/ \
<<<<<<< HEAD
            --test_file data/{dataset}/{dataset}.csv \
=======
            --test_file data/{dataset}/{dataset}_test.csv \
>>>>>>> ebf225522d74eee662e869d6d7f5ba0aca04a277
            --out_file results/{experiment}/preds_{name}_on_{dataset}.txt \
            --ref_file data/reference/reference_{dataset}_full.txt \
            --result_file results/{experiment}/rouge_{name}_on_{dataset}.csv")

datasets = ['tldr', 'small_billsum']
name = 'tosdr_full'
checkpoint = 'tosdr_batchsize16_lr3e-05_seed161_full_datasetTrue/checkpoint-8'
for dataset in datasets:
    os.system(f"python evaluate_bart.py bart-large-cnn-finetuned/{checkpoint}/ \
<<<<<<< HEAD
            --test_file data/{dataset}/{dataset}.csv \
=======
            --test_file data/{dataset}/{dataset}_test.csv \
>>>>>>> ebf225522d74eee662e869d6d7f5ba0aca04a277
            --out_file results/{experiment}/preds_{name}_on_{dataset}.txt \
            --ref_file data/reference/reference_{dataset}_full.txt \
            --result_file results/{experiment}/rouge_{name}_on_{dataset}.csv")

datasets = ['tosdr', 'tldr']
name = 'small_billsum_full'
checkpoint = 'small_billsum_batchsize16_lr3e-05_seed161_full_datasetTrue/checkpoint-120'
for dataset in datasets:
    os.system(f"python evaluate_bart.py bart-large-cnn-finetuned/{checkpoint}/ \
<<<<<<< HEAD
            --test_file data/{dataset}/{dataset}.csv \
=======
            --test_file data/{dataset}/{dataset}_test.csv \
>>>>>>> ebf225522d74eee662e869d6d7f5ba0aca04a277
            --out_file results/{experiment}/preds_{name}_on_{dataset}.txt \
            --ref_file data/reference/reference_{dataset}_full.txt \
            --result_file results/{experiment}/rouge_{name}_on_{dataset}.csv")


