import os
'''EXAMPLE
dataset = tldr
name = tldr_test
experiment = TODO
checkpoint = checkpoint-1

os.system(f"python evaluate_bart.py bart-large-cnn-finetuned/{checkpoint}/ \
            --test_file data/{dataset}/{name}.csv \
            --out_file results/{experiment}/preds_{name}.txt \
            --ref_file data/reference/reference_{name}.txt \
            --result_file results/{experiment}/rouge_{name}.csv")
'''
