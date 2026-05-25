from transformers import AutoTokenizer
from src.config import MODEL_NAME, MAX_LENGTH, SEED

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def preprocess_function(examples):
    inputs = [example["en"] for example in examples["translation"]]
    targets = [example["de"] for example in examples["translation"]]
    model_inputs = tokenizer(inputs, max_length=MAX_LENGTH, truncation=True)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=MAX_LENGTH, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def tokenize_and_split(ds):
    tokenized = ds.map(preprocess_function, batched=True)
    train_val = tokenized.train_test_split(test_size=0.2, seed=SEED)
    val_test = train_val["test"].train_test_split(test_size=0.5, seed=SEED)
    return train_val["train"], val_test["train"], val_test["test"]