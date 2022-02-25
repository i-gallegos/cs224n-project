import os
import pandas as pd
from transformers import AutoModel, AutoTokenizer, DataCollatorForSeq2Seq
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import Seq2SeqTrainingArguments, TrainingArguments, Trainer
from transformers import TrainerCallback, EarlyStoppingCallback
from tqdm.notebook import tqdm
from torch.utils.data import DataLoader, Dataset
from datasets import load_metric, load_dataset

# Download the model
model_name = "facebook/bart-large-cnn"
tokenizer = AutoTokenizer.from_pretrained(model_name)
metric = load_metric("rouge")

BATCH_SIZE = 16
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


def create_datasets(train_path, dev_path, test_path):
    data_files = {}
    data_files["train"] = train_path
    data_files["validation"] = dev_path
    data_files["test"] = test_path
    raw_datasets = load_dataset('csv', data_files=data_files)
    return raw_datasets

def train(raw_datasets):
    # Load model
    model = AutoModel.from_pretrained(model_name)
    train_dataset = raw_datasets["train"].map(preprocess_function, batched=True)
    dev_dataset = raw_datasets["validation"].map(preprocess_function, batched=True)

    # Fine-tuning parameters
    arguments = Seq2SeqTrainingArguments(
        output_dir=f"{model_name}-finetuned",
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=3,
        evaluation_strategy="epoch", # run validation at the end of each epoch
        save_strategy="epoch",
        learning_rate=2e-5,
        load_best_model_at_end=True,
        seed=224
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # Train the model
    trainer = Trainer(
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

    # Evaluate the model
    results = trainer.predict(small_tokenized_dataset['val']) # also gives you predictions

def test(test_text):
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
