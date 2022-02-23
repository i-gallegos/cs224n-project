"""
HOW TO USE ROUGE (for datasets)
install library through pip: pip install rouge

[The get_scores method returns three metrics, ROUGE-N using a unigram (ROUGE-1) and a bigram (ROUGE-2) — and ROUGE-L.
For each of these, we receive the F1 score f, precision p, and recall r.]
"""

from rouge import Rouge
import argparse

# outputs rouge scores
def eval(model_file, ref_file):
    # get text from given files
    with open(model_file, "r") as mf:
        model_txt = mf.read()
    with open(ref_file, "r") as rf:
        ref_txt = rf.read()

    assert(len(model_txt)) != 0
    assert(len(ref_txt)) != 0

    model_txt = model_txt.split('\n')
    ref_txt = ref_txt.split('\n')

    # compute rouge scores
    rouge = Rouge()
    scores = rouge.get_scores(model_txt, ref_txt, avg=True)
    print("Rouge scores: ")
    print(scores)
    return scores

# driver code: parse model_output and reference 
def eval_rogue(args):
    argp = argparse.ArgumentParser()
    argp.add_argument('model_output', help="predicted summarized sentences model outputs")
    argp.add_argument('reference_text', help = "reference summarized sentences")
    args = argp.parse_args()
    model_file = args.model_output
    ref_file = args.reference_text

    # run rouge evaluation
    eval(model_file, ref_file)
