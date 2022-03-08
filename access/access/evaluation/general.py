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

from easse.cli import evaluate_system_output

from access.preprocess import lowercase_file, to_lrb_rrb_file
from access.resources.paths import get_data_filepath, get_law_filepath, get_pred_filepath, get_post_filepath, get_pred_filepath_post, get_pre_summarization
from access.utils.helpers import mute, get_temp_filepath

'''A simplifier is a method with signature: simplifier(complex_filepath, output_pred_filepath)'''


def get_prediction_on_turkcorpus(simplifier, phase):
    source_filepath = get_data_filepath('turkcorpus', phase, 'complex')
    pred_filepath = get_temp_filepath()
    with mute():
        simplifier(source_filepath, pred_filepath)
    return pred_filepath

"""
Here, uses BLEU, SARI, and FKGL metrics to evaluate
"""

def evaluate_simplifier_on_turkcorpus(simplifier, phase):
    pred_filepath = get_prediction_on_turkcorpus(simplifier, phase)
    pred_filepath = lowercase_file(pred_filepath)
    pred_filepath = to_lrb_rrb_file(pred_filepath)
    return evaluate_system_output(f'turkcorpus_{phase}_legacy',
                                  sys_sents_path=pred_filepath,
                                  metrics=['bleu', 'sari_legacy', 'fkgl'],
                                  quality_estimation=True)

"""
TODO: add any functions for eval for our specific datasets
"""
def get_prediction_on_law(dataset, simplifier, phase):
    orig_filepath, reference_filepath = get_law_filepath(dataset, phase)
    #orig_filepath, reference_filepath = get_pre_summarization(dataset)
    pred_filepath = get_pred_filepath(dataset, phase)
    with mute():
        simplifier(orig_filepath, pred_filepath)
    return pred_filepath, reference_filepath, orig_filepath

def evaluate_simplifier_on_law(dataset, simplifier, phase):
    pred_filepath, reference_filepath, orig_filepath = get_prediction_on_law(dataset, simplifier, phase)
    #pred_filepath = lowercase_file(pred_filepath)
    #pred_filepath = to_lrb_rrb_file(pred_filepath)
    return evaluate_system_output(test_set='custom', 
                                  sys_sents_path= pred_filepath,
                                  orig_sents_path = orig_filepath,
                                  refs_sents_paths =[reference_filepath],
                                  metrics=['bleu', 'sari_legacy', 'fkgl'],
                                  quality_estimation=True)

# for post BART -- slightly different fetching mechanism
def get_prediction_post(dataset, simplifier, phase, sum_model):
    orig_filepath, reference_filepath = get_post_filepath(dataset, phase, sum_model)
    pred_filepath = get_pred_filepath_post(dataset, phase, sum_model)
    with mute():
        simplifier(orig_filepath, pred_filepath)
    return pred_filepath, reference_filepath, orig_filepath

def evaluate_simplifier_on_law_post(dataset, simplifier, phase, sum_model): 
    pred_filepath, reference_filepath, orig_filepath = get_prediction_post(dataset, simplifier, phase, sum_model)
    pred_filepath = lowercase_file(pred_filepath)
    pred_filepath = to_lrb_rrb_file(pred_filepath)
    return evaluate_system_output(test_set='custom', 
                                  sys_sents_path=pred_filepath,
                                  orig_sents_path = orig_filepath,
                                  refs_sents_paths =[reference_filepath],
                                  metrics=['bleu', 'sari_legacy', 'fkgl'],
                                  quality_estimation=True)
