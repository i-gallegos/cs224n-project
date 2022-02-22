# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from access.evaluation.general import evaluate_simplifier_on_turkcorpus
from access.preprocessors import get_preprocessors
from access.resources.prepare import prepare_turkcorpus, prepare_models
from access.simplifiers import get_fairseq_simplifier, get_preprocessed_simplifier

"""
TODO: Change evaluation methods where needed.
SARI and FKGL evaluation for simplification.

HOW TO USE ROUGE (for datasets)
install library through pip: pip install rouge

[in the code, do following]
from rouge import Rouge
model_output = ["string", "s1", etc]
reference = ["string", "s1", etc]
rouge = Rouge()
rouge.get_scores(model_output, reference, avg=True)

[The get_scores method returns three metrics, ROUGE-N using a unigram (ROUGE-1) and a bigram (ROUGE-2) — and ROUGE-L.
For each of these, we receive the F1 score f, precision p, and recall r.]
"""


if __name__ == '__main__':
    print('Evaluating pretrained model')
    prepare_turkcorpus()
    best_model_dir = prepare_models()
    recommended_preprocessors_kwargs = {
        'LengthRatioPreprocessor': {'target_ratio': 0.95},
        'LevenshteinPreprocessor': {'target_ratio': 0.75},
        'WordRankRatioPreprocessor': {'target_ratio': 0.75},
        'SentencePiecePreprocessor': {'vocab_size': 10000},
    }
    preprocessors = get_preprocessors(recommended_preprocessors_kwargs)
    simplifier = get_fairseq_simplifier(best_model_dir, beam=8)
    simplifier = get_preprocessed_simplifier(simplifier, preprocessors=preprocessors)
    print(evaluate_simplifier_on_turkcorpus(simplifier, phase='test'))
