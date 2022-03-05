from transformers import pipeline, AutoTokenizer
import pandas as pd
import random
import argparse
import evalRouge
random.seed(0)

argp = argparse.ArgumentParser()
argp.add_argument('model_path', help="bart-large-cnn-finetuned/<model-checkpoint>/")
argp.add_argument('--test_file', help="data/<dataset>/<name>.csv", default=None)
argp.add_argument('--out_file', help="results/<experiment>/preds_<dataset>.txt", default=None)
argp.add_argument('--ref_file', help="data/<dataset>/<name>.txt", default=None)
argp.add_argument('--result_file', help="results/<experiment>/rouge_<dataset>.csv", default=None)
args = argp.parse_args()

MAX_SOURCE_LENGTH = 128
MAX_TARGET_LENGTH = 64
PADDING = "max_length"

def test_file_to_documents():
    df = pd.read_csv(args.test_file)
    document = df['document'].tolist()

    # Truncate for BART model
    document = [(' ').join(d.split()[:512]) for d in document]
    return document

def evaluate():
    inputs = test_file_to_documents()
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn", model_max_length=MAX_SOURCE_LENGTH, padding=PADDING, truncation=True)
    summarizer = pipeline("summarization", model=args.model_path, config=(args.model_path+"/config.json"), device=0)
    outputs = summarizer(inputs, max_length=MAX_TARGET_LENGTH, do_sample=False)
    outputs = [output['summary_text'] for output in outputs]

    with open(args.out_file, 'w') as f:
        for output in outputs:
            f.write("%s\n" % output)

def compute_rouge():
    df = pd.DataFrame(columns=['R-1', 'R-2', 'R-L'])

    preds = args.out_file
    true = args.ref_file

    rouge = evalRouge.eval(preds, true)
    df = pd.DataFrame.from_dict({'R-1':[rouge['rouge-1']['f']],
                                 'R-2':[rouge['rouge-2']['f']],
                                 'R-L':[rouge['rouge-l']['f']]})

    df.to_csv(args.result_file, index=False)

def main():
    evaluate()
    compute_rouge()

if __name__ == "__main__":
    main()