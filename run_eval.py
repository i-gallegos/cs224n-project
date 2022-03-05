import os
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
