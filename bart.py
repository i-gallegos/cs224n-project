from transformers import AutoModel, AutoTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import TrainingArguments, Trainer
from transformers import TrainerCallback, EarlyStoppingCallback
from tqdm.notebook import tqdm
from torch.utils.data import DataLoader

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

# Download the model
model_name = "facebook/bart-large-cnn"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Create dataloaders
# TODO

def train():
    # Load model
    model = AutoModel.from_pretrained(model_name)

    # Fine-tuning parameters
    arguments = TrainingArguments(
        output_dir="trainer",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
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
        train_dataset=small_tokenized_dataset['train'],
        eval_dataset=small_tokenized_dataset['val'], # change to test when you do your final evaluation!
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
    print(["NEGATIVE", "POSITIVE"][prediction])

def main():
    train()
    test()

if __name__ == "__main__":
    main()
