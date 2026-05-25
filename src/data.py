from datasets import load_dataset
from src.config import SEED, DATASET_SIZE

def load_data():
    ds = load_dataset("Helsinki-NLP/europarl", "de-en")
    return ds["train"].shuffle(seed=SEED).select(range(DATASET_SIZE))

def clean_data(ds):
    clean_ds = ds.filter(lambda x: 
        len(x["translation"]["en"].split()) >= 3 and 
        len(x["translation"]["de"].split()) >= 3 and
        len(x["translation"]["en"].split()) <= 128 and
        len(x["translation"]["de"].split()) <= 128 and
        0.5 <= len(x["translation"]["de"].split()) / len(x["translation"]["en"].split()) <= 2.0
    )
    seen = set()
    def remove_duplicates(example):
        en = example["translation"]["en"]
        if en in seen:
            return False
        seen.add(en)
        return True
    return clean_ds.filter(remove_duplicates)