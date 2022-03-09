import random
import os
import sys
import re
import pandas as pd
from summa.summarizer import summarize
from sumy.summarizers.kl import KLSummarizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from transformers import pipeline
import evalRouge

DATASETS = ['tldr', 'tosdr', 'small_billsum']
# SPLITS = ['train', 'dev', 'test']
SPLITS = ['test']

def adjust_summary_len(summary, avg_summary_len):
    '''For the sentence which causes the summary to exceed the budget, we keep
    or discard the full sentence depending on which resulting summary is closer
    to the budgeted length.
    '''
    tokenized = summary.split('.')
    total_len = len(summary.split())
    while len(tokenized) > 1 and total_len > avg_summary_len:
        if abs(avg_summary_len - total_len) > abs(avg_summary_len - (total_len - len(tokenized[-1].split()))):
            total_len -= len(tokenized[-1].split())
            tokenized.pop()
            print(total_len, tokenized)
        else:
            break

    return (' ').join([str(sentence) for sentence in tokenized])


def text_rank(text, avg_summary_len):
    '''Harnesses the PageRank algorithm to choose the sentences with the
    highest similarity scores to the original document.

    Source: https://github.com/summanlp/textrank
    '''
    summary = summarize(text, words=avg_summary_len, ratio=1)
    if len(summary.split()) > avg_summary_len:
        summary = adjust_summary_len(summary, avg_summary_len)
    if len(summary) == 0:
        summary += 'none'
    return summary.replace('\n', ' ')


def kl_sum(text, avg_summary_len):
    '''Greedily selects the sentences that minimize the Kullback-Lieber (KL) divergence
    between the original text and proposed summary.

    Sources: http://nlp.cs.berkeley.edu/pubs/Haghighi-Vanderwende_2009_Summarization_paper.pdf
             https://pypi.org/project/sumy/
    '''
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = KLSummarizer()

    summary = summarizer(parser.document, 1)
    for i in range(2, len(text.split('.'))):
        prev_len = len((' ').join([str(sentence) for sentence in summary]).split())
        new_summary = summarizer(parser.document, i)
        new_len = len((' ').join([str(sentence) for sentence in new_summary]).split())
        if abs(avg_summary_len - new_len) >= abs(avg_summary_len - new_len):
            break
        else:
            summary = new_summary

    if len(summary) == 0:
        summary = 'none'
    return (' ').join([str(sentence) for sentence in summary])


def lead_one(text):
    '''Select the first sentence of the original text as the summary.
    '''
    return text.split('.')[0]


def lead_k(text, avg_summary_len):
    '''
    Selects the first k sentences until a word limit is satisfied.
    '''
    tokenized = text.split('.')

    summary = [tokenized[0]]
    total_len = len(tokenized[0].split())

    for i in range(1, len(tokenized)):
        len_i = len(tokenized[i].split())
        if abs(avg_summary_len - total_len) > abs(avg_summary_len - (total_len+len_i)):
            summary.append(tokenized[i])
            total_len += len_i
        else:
            break

    if len(summary) == 0:
        summary += 'none'
    return ('. ').join(summary)


def random_k(text, avg_summary_len):
    '''Selects a random sentence until a word limit is satisfied. For this baseline,
    the reported numbers are an average of 10 runs on the entire dataset.
    '''
    tokenized = text.split('.')

    random_shuffle = list(range(len(tokenized)))
    random.shuffle(random_shuffle)

    i = random_shuffle.pop()
    summary = [tokenized[i]]
    total_len = len(tokenized[i].split())

    while random_shuffle:
        i = random_shuffle.pop()
        len_i = len(tokenized[i].split())
        if abs(avg_summary_len - total_len) > abs(avg_summary_len - (total_len+len_i)):
            summary.append(tokenized[i])
            total_len += len_i
        else:
            break

    if len(summary) == 0:
        summary += 'none'
    return ('. ').join(summary)


def bart_no_finetuning(text, summarizer, avg_summary_len):
    text = (' ').join(text.split()[:512])
    try:
        summary = summarizer(text, max_length=min(1024, avg_summary_len+10), min_length=10, do_sample=False)[0]['summary_text']
    except:
        summary = 'none'
    return summary

def baseline_summaries(dataset, split, filepath, summarizer, simplified):
    df = pd.read_csv(filepath)

    # avg_summary_len should be average number of words among all summaries
    avg_summary_len = int(df['summary'].str.split().apply(len).mean())

    if simplified == 'pre':
        out_dir = os.path.join('results', 'baselines', 'simplified', dataset, split)
    else:
        out_dir = os.path.join('results', 'baselines', dataset, split)
    os.makedirs(out_dir, exist_ok=True)

    fout_text_rank = os.path.join(out_dir, 'text_rank.txt')
    fout_kl_sum = os.path.join(out_dir, 'kl_sum.txt')
    fout_lead_one = os.path.join(out_dir, 'lead_one.txt')
    fout_lead_k = os.path.join(out_dir, 'lead_k.txt')
    fout_random_k = os.path.join(out_dir, 'random_k.txt')
    fout_bart = os.path.join(out_dir, 'bart.txt')
    fout_ref = os.path.join(out_dir, 'ref.txt')

    fo_text_rank = open(fout_text_rank, 'w')
    fo_kl_sum = open(fout_kl_sum, 'w')
    fo_lead_one = open(fout_lead_one, 'w')
    fo_lead_k = open(fout_lead_k, 'w')
    fo_random_k = open(fout_random_k, 'w')
    fo_bart = open(fout_bart, 'w')
    fo_ref = open(fout_ref, 'w')

    for index, row in df.iterrows():
        print(f'  {index} of {len(df)}')
        document = row['document']
        summary = row['summary']

        fo_text_rank.write(text_rank(document, avg_summary_len).strip() + '\n')
        fo_kl_sum.write(kl_sum(document, avg_summary_len).strip() + '\n')
        fo_lead_one.write(lead_one(document).strip() + '\n')
        fo_lead_k.write(lead_k(document, avg_summary_len).strip() + '\n')
        fo_random_k.write(random_k(document, avg_summary_len).strip() + '\n')
        fo_bart.write(bart_no_finetuning(document, summarizer, avg_summary_len).strip() + '\n')
        fo_ref.write(summary.strip() + '\n')

    fo_text_rank.close()
    fo_kl_sum.close()
    fo_lead_one.close()
    fo_lead_k.close()
    fo_random_k.close()
    fo_bart.close()
    fo_ref.close()


def run_baselines(simplified='none'):
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=0) # for GPU
    # summarizer = None

    for dataset in DATASETS:
        dir =  os.path.join('data', dataset)
        for split in SPLITS:
            if simplified == 'pre':
                filepath = os.path.join(dir, dataset+'_'+split+'_simplified.csv')
            else:
                filepath = os.path.join(dir, dataset+'_'+split+'.csv')
            print(f'PROCESSING {filepath}')
            baseline_summaries(dataset, split, filepath, summarizer, simplified)


def compute_metrics(simplified='none'):
    df = pd.DataFrame(columns=['dataset', 'split', 'baseline', 'R-1', 'R-2', 'R-L'])
    for dataset in DATASETS:
        if simplified == 'pre':
            dir =  os.path.join('results', 'baselines', 'simplified', dataset)
        elif simplified == 'post':
            dir_sum = os.path.join('access', 'access', 'preds', dataset)
            dir_ref = os.path.join('results', 'baselines', dataset)
        else:
            dir = os.path.join('results', 'baselines', dataset)

        for split in SPLITS:
            for baseline in ['bart', 'kl_sum', 'lead_k', 'lead_one', 'random_k', 'text_rank']:
                print(f'Computing metrics for {dataset}, {split}, {baseline}')

                if simplified == 'post':
                    preds = os.path.join(dir_sum, 'preds_'+dataset+'_'+split+'_'+baseline)
                    true = os.path.join(dir_ref, split, 'ref.txt')
                else:
                    preds = os.path.join(dir, split, baseline+'.txt')
                    true = os.path.join(dir, split, 'ref.txt')

                print(preds, true)

                rouge = evalRouge.eval(preds, true)
                print(rouge['rouge-1']['f'], rouge['rouge-2']['f'], rouge['rouge-l']['f'])
                df = pd.concat((df, pd.DataFrame.from_dict({'dataset':dataset,
                                                            'split':split,
                                                            'baseline':baseline,
                                                            'R-1':[rouge['rouge-1']['f']],
                                                            'R-2':[rouge['rouge-2']['f']],
                                                            'R-L':[rouge['rouge-l']['f']]})), ignore_index=True)

    if simplified == 'pre':
        save_path = os.path.join('results', 'baselines', 'baseline_simplified_rouge.csv')
    elif simplified == 'post':
        save_path = os.path.join('results', 'baselines', 'baseline_simplified_post_rouge.csv')
    else:
        save_path = os.path.join('results', 'baselines', 'baseline_rouge.csv')

    df.to_csv(save_path, index=False)

def main():
    run_baselines(simplified='pre')
    compute_metrics(simplified='pre')


if __name__ == "__main__":
    main()
