import os
import pandas as pd
from transformers import AutoModel, AutoTokenizer, DataCollatorForSeq2Seq
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import Seq2SeqTrainingArguments, TrainingArguments, Trainer, Seq2SeqTrainer
from transformers import TrainerCallback, EarlyStoppingCallback
from tqdm.notebook import tqdm
from torch.utils.data import DataLoader, Dataset
import torch
from datasets import load_metric, load_dataset

device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
num_workers = 0 if device == 'cpu' else 4

# Download the model
model_name = "facebook/bart-large-cnn"
tokenizer = AutoTokenizer.from_pretrained(model_name)
metric = load_metric("rouge")

BATCH_SIZE = 1
NUM_TRAIN_EPOCHS = 3
LEARNING_RATE = 2e-5
MAX_SOURCE_LENGTH = 1024
MAX_TARGET_LENGTH = 128
PADDING = "max_length"

class LoggingCallback(TrainerCallback):
    def __init__(self, log_path):
        self.log_path = log_path

    def on_log(self, args, state, control, logs=None, **kwargs):
        _ = logs.pop("total_flos", None)
        if state.is_local_process_zero:
            with open(self.log_path, "a") as f:
                f.write(json.dumps(logs) + "\n")

def create_datasets(train_path, dev_path, test_path):
    data_files = {}
    data_files["train"] = train_path
    data_files["validation"] = dev_path
    data_files["test"] = test_path
    raw_datasets = load_dataset('csv', data_files=data_files)
    print(type(raw_datasets['train']['summary']))
    print(raw_datasets['train']['id'][0])
    raw_datasets = load_dataset('xsum')
    print(type(raw_datasets['train']['summary']))
    print(raw_datasets['train']['id'][0])
    exit(0)
    return raw_datasets


def preprocess_function(examples):
    inputs, targets = [], []
    for i in range(len(examples["document"])):
        if examples["document"][i] is not None and examples["summary"][i] is not None:
            inputs.append(examples["document"][i])
            targets.append(examples["summary"][i])

    model_inputs = tokenizer(inputs, max_length=MAX_SOURCE_LENGTH, padding=PADDING, truncation=True)

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=MAX_TARGET_LENGTH, padding=PADDING, truncation=True)

    # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
    # padding in the loss.
    if PADDING == "max_length":
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        ]

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    if data_args.ignore_pad_token_for_loss:
        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    # Extract a few results from ROUGE
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result


def train(raw_datasets):
    # Load model
    model = AutoModel.from_pretrained(model_name).to(device)
    train_dataset = raw_datasets["train"].map(preprocess_function, batched=True)
    dev_dataset = raw_datasets["validation"].map(preprocess_function, batched=True)

    # Fine-tuning parameters
    arguments = Seq2SeqTrainingArguments(
        output_dir=f"{model_name}-finetuned",
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        evaluation_strategy="epoch", # run validation at the end of each epoch
        save_strategy="epoch",
        learning_rate=LEARNING_RATE,
        load_best_model_at_end=True,
        seed=224
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, label_pad_token_id=-100)

    # Train the model
    trainer = Seq2SeqTrainer(
        model=model,
        args=arguments,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=1, early_stopping_threshold=0.0))
    trainer.add_callback(LoggingCallback("trainer/log.jsonl"))
    print("TRAINING")
    trainer.train()

    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model()  # Saves the tokenizer too for easy upload

    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

def evaluate(test_text):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    finetuned_model = AutoModel.from_pretrained("trainer/checkpoint-24")
    model_inputs = tokenizer(test_text, return_tensors="pt")

    # TODO: change for summary task
    prediction = torch.argmax(finetuned_model(**model_inputs).logits)


def main():
    dataset = 'tldr'
    train_path = os.path.join('data', dataset, dataset+'_train.csv')
    dev_path = os.path.join('data', dataset, dataset+'_dev.csv')
    test_path = os.path.join('data', dataset, dataset+'_test.csv')
    raw_datasets = create_datasets(train_path, dev_path, test_path)

    train(raw_datasets)
    # test()


if __name__ == "__main__":
    main()
