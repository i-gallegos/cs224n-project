from transformers import AutoModel, AutoTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import TrainingArguments, Trainer
from transformers import TrainerCallback, EarlyStoppingCallback
from tqdm.notebook import tqdm
from torch.utils.data import DataLoader

# Download the model
model_name = "facebook/bart-large-cnn"
tokenizer = AutoTokenizer.from_pretrained(model_name)

BATCH_SIZE = 16

class CustomDataset(Dataset):
  def __init__(self,file_name):
    df=pd.read_csv(file_name)

    x=df['original_text'].values
    y=df['reference_summary'].values

    self.x=x
    self.y=y

  def __len__(self):
    return len(self.y)

  def __getitem__(self,idx):
    return self.x[idx],self.y[idx]


class LoggingCallback(TrainerCallback):
    def __init__(self, log_path):
        self.log_path = log_path

    def on_log(self, args, state, control, logs=None, **kwargs):
        _ = logs.pop("total_flos", None)
        if state.is_local_process_zero:
            with open(self.log_path, "a") as f:
                f.write(json.dumps(logs) + "\n")

def compute_metrics(eval_pred):
    """Called at the end of validation. Gives accuracy"""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    # calculates the accuracy
    return {"accuracy": np.mean(predictions == labels)}

def create_dataloaders(train_path, dev_path, test_path):
    train_dataset = CustomDataset(train_path)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)

    dev_dataset = CustomDataset(dev_path)
    dev_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE, shuffle=False)

    test_dataset = CustomDataset(test_path)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, dev_loader, test_loader

def train(train_dataset, dev_dataset):
    # Load model
    model = AutoModel.from_pretrained(model_name)

    # Fine-tuning parameters
    arguments = TrainingArguments(
        output_dir="trainer",
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=3,
        evaluation_strategy="epoch", # run validation at the end of each epoch
        save_strategy="epoch",
        learning_rate=2e-5,
        load_best_model_at_end=True,
        seed=224
    )

    # Train the model
    trainer = Trainer(
        model=model,
        args=arguments,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=1, early_stopping_threshold=0.0))
    trainer.add_callback(LoggingCallback("trainer/log.jsonl"))
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
    train_loader, dev_loader, test_loader = create_dataloaders(train_path, dev_path, test_path)

    train(train_loader, dev_loader)
    # test()


if __name__ == "__main__":
    main()
