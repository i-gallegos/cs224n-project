import os
import pandas as pd
import numpy as np
import nltk
import json
from transformers import AutoTokenizer, DataCollatorForSeq2Seq, AutoModelForSeq2SeqLM
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import TrainerCallback, EarlyStoppingCallback
from tqdm.notebook import tqdm
import torch
import wandb
from datasets import load_metric, load_dataset

device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'

model_name = "facebook/bart-large-cnn"
tokenizer = AutoTokenizer.from_pretrained(model_name)
metric = load_metric("rouge")

TRAIN_ON_FULL_DATASET = True # when TRAIN_ON_FULL_DATASET=True: training data = 'train_and_dev; validation data = 'test'
DATASET = 'tldr'
BATCH_SIZE = 16
NUM_TRAIN_EPOCHS = 1
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
GRAD_ACCUMULATION_STEPS = 4
SEED = 224
MAX_SOURCE_LENGTH = 128
MAX_TARGET_LENGTH = 64
PADDING = "max_length"

with open("wandb_token.txt", "r") as f:
    token=f.readlines()[0].rstrip()
wandb.login(key=token)

class LoggingCallback(TrainerCallback):
    def __init__(self, log_path):
        self.log_path = log_path

    def on_log(self, args, state, control, logs=None, **kwargs):
        _ = logs.pop("total_flos", None)
        if state.is_local_process_zero:
            with open(self.log_path, "a") as f:
                f.write(json.dumps(logs) + "\n")


def create_datasets(train_path, dev_path, test_path):
    '''Create custom dataset from .csv files with columns [document, summary, ids].
    '''
    data_files = {}
    data_files["train"] = train_path
    data_files["validation"] = dev_path
    data_files["test"] = test_path
    raw_datasets = load_dataset('csv', data_files=data_files)
    return raw_datasets


def preprocess_function(examples):
    '''Tokenize document and summary to prepare input for model.
    '''
    prefix = ""
    inputs = [prefix + doc for doc in examples["document"]]
    model_inputs = tokenizer(inputs, max_length=MAX_SOURCE_LENGTH, padding=PADDING, truncation=True)

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["summary"], max_length=MAX_TARGET_LENGTH, padding=PADDING, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Rouge expects a newline after each sentence
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    # Extract a few results
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

    # Add mean generated length
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)

    return {k: round(v, 4) for k, v in result.items()}


def train(tokenized_datasets):
    # Load model
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

    # Fine-tuning parameters
    args = Seq2SeqTrainingArguments(
        "bart-large-cnn-finetuned",
        evaluation_strategy = "epoch",
        save_strategy="epoch",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        weight_decay=WEIGHT_DECAY,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        gradient_accumulation_steps=GRAD_ACCUMULATION_STEPS,
        predict_with_generate=True,
        load_best_model_at_end=True,
        seed=SEED,
        report_to="wandb",
        run_name=DATASET+'_batchsize='+str(BATCH_SIZE)+'_lr='+str(LEARNING_RATE)+'_seed='+str(SEED)
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # Train the model
    trainer = Seq2SeqTrainer(
      model,
      args,
      train_dataset=tokenized_datasets["train"],
      eval_dataset=tokenized_datasets["validation"],
      data_collator=data_collator,
      tokenizer=tokenizer,
      compute_metrics=compute_metrics
    )
    trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=1, early_stopping_threshold=0.0))
    trainer.add_callback(LoggingCallback("bart-large-cnn-finetuned/log.json"))
    trainer.train()
    trainer.save_model()

    with torch.no_grad():
        train_metrics = trainer.predict(tokenized_datasets["train"]).metrics
        val_metrics = trainer.predict(tokenized_datasets["validation"]).metrics
        trainer.log_metrics("bart-large-cnn-finetuned/train", train_metrics)
        trainer.save_metrics("bart-large-cnn-finetuned/train", train_metrics)
        trainer.log_metrics("bart-large-cnn-finetuned/val", val_metrics)
        trainer.save_metrics("bart-large-cnn-finetuned/vali", val_metrics)

    wandb.finish()

def evaluate(test_text): # TODO: do not use as is
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    finetuned_model = AutoModel.from_pretrained("trainer/checkpoint-24")
    model_inputs = tokenizer(test_text, return_tensors="pt")


def main():
    if not TRAIN_ON_FULL_DATASET:
        train_path = os.path.join('data', DATASET, DATASET+'_train.csv')
        dev_path = os.path.join('data', DATASET, DATASET+'_dev.csv')
        test_path = os.path.join('data', DATASET, DATASET+'_test.csv')
    else:
        train_path = os.path.join('data', DATASET, DATASET+'_train_and_dev.csv')
        dev_path = os.path.join('data', DATASET, DATASET+'_test.csv')
        test_path = os.path.join('data', DATASET, DATASET+'_test.csv')

    raw_datasets = create_datasets(train_path, dev_path, test_path)
    tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)
    train(tokenized_datasets)

if __name__ == "__main__":
    main()
