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

class CustomDataset(Dataset):
    def __init__(self,file_name):
        df = pd.read_csv(file_name)
        x = tokenizer(df['original_text'].tolist(), truncation=True, padding=True)
        with tokenizer.as_target_tokenizer():
            y = tokenizer(df['reference_summary'].tolist(), truncation=True, padding=True)
        id = range(len(y))


        self.document = df['original_text'].tolist()
        self.summary = df['reference_summary'].tolist()
        self.id = id
        self.input_ids = x['input_ids']
        self.attention_mask = x['attention_mask']
        self.labels = y['input_ids']

    def __len__(self):
        return len(self.id)

    def __getitem__(self,idx):
        return {'document':self.document[idx],
                'summary':self.summary[idx],
                'id':self.id[idx],
                'input_ids':self.input_ids[idx],
                'attention_mask':self.attention_mask[idx],
                'labels':self.labels[idx]}


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


def create_datasets(train_path, dev_path, test_path):
    prefix = "summarize: "
    # prefix = ""
    max_input_length = 1024
    max_target_length = 128


    def preprocess_function(examples):
        inputs = [prefix + doc for doc in examples["document"]]
        model_inputs = tokenizer(inputs, padding=True, truncation=True)
        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(examples["summary"], padding=True, truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    train_dataset = CustomDataset(train_path)
    dev_dataset = CustomDataset(dev_path)
    test_dataset = CustomDataset(test_path)

    # raw_datasets = load_dataset("xsum")
    # train_dataset = raw_datasets['train'].map(preprocess_function, batched=True)
    # dev_dataset = raw_datasets.map(preprocess_function, batched=True)
    # test_dataset = raw_datasets.map(preprocess_function, batched=True)


    # return train_loader, dev_loader, test_loader
    return train_dataset, dev_dataset, test_dataset

def train(train_dataset, dev_dataset):
    # Load model
    model = AutoModel.from_pretrained(model_name)

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
    dataset = 'tosdr'
    train_path = os.path.join('data', dataset, dataset+'_train.csv')
    dev_path = os.path.join('data', dataset, dataset+'_dev.csv')
    test_path = os.path.join('data', dataset, dataset+'_test.csv')
    train_loader, dev_loader, test_loader = create_datasets(train_path, dev_path, test_path)

    train(train_loader, dev_loader)
    # test()


if __name__ == "__main__":
    main()
