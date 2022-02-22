"""
HOW TO USE ROUGE (for datasets)
install library through pip: pip install rouge

[The get_scores method returns three metrics, ROUGE-N using a unigram (ROUGE-1) and a bigram (ROUGE-2) — and ROUGE-L.
For each of these, we receive the F1 score f, precision p, and recall r.]
"""

from rouge import Rouge

# load in model_output and reference with appropriate files
model_output = ["string"]
reference = ["str"]

rouge = Rouge()
scores = rouge.get_scores(model_output, reference, avg=True)
print("Rouge scores: ")
print(scores)
