# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import sys
scripts_dir = os.path.dirname(__file__)
access_dir = os.path.join(scripts_dir, '..')
sys.path.append(access_dir)

from access.evaluation.general import evaluate_simplifier_on_law
from access.preprocessors import get_preprocessors
from access.resources.prepare import prepare_turkcorpus, prepare_models
from access.simplifiers import get_fairseq_simplifier, get_preprocessed_simplifier

"""
TODO: Change evaluation methods where needed.
SARI and FKGL evaluation for simplification.
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
    #print(evaluate_simplifier_on_turkcorpus(simplifier, phase='test'))
    print("tldr (test)")
    print(evaluate_simplifier_on_law('tldr', simplifier, phase='test'))
    #print("tosdr (test)")
    #print(evaluate_simplifier_on_law('tosdr', simplifier, phase='test'))
    print("billsum (test)")
    print(evaluate_simplifier_on_law('small_billsum', simplifier, phase='test'))
    