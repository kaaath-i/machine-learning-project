import evaluate as hf_evaluate
import numpy as np
from nltk.translate.meteor_score import meteor_score
from transformers import pipeline
from src.config import NLLB_MODEL, MODEL_NAME, MAX_LENGTH, BATCH_SIZE
from src.preprocessing import tokenizer

bleu = hf_evaluate.load("sacrebleu")
chrf = hf_evaluate.load("chrf")


def compute_meteor(predictions, references):
    scores = [
        meteor_score([ref.split()], pred.split())
        for pred, ref in zip(predictions, references)
    ]
    return sum(scores) / len(scores)


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels_bleu = [[label.strip()] for label in decoded_labels]
    decoded_labels_flat = [label.strip() for label in decoded_labels]

    return {
        "bleu": bleu.compute(predictions=decoded_preds, references=decoded_labels_bleu)["score"],
        "meteor": compute_meteor(decoded_preds, decoded_labels_flat),
        "chrf": chrf.compute(predictions=decoded_preds, references=decoded_labels_bleu)["score"],
    }


def evaluate_nllb(test_sources, test_references):
    translator = pipeline("translation", model=NLLB_MODEL, device=0)
    results = translator(test_sources, src_lang="eng_Latn", tgt_lang="deu_Latn", max_length=MAX_LENGTH, batch_size=BATCH_SIZE)
    predictions = [r["translation_text"] for r in results]
    refs_bleu = [[r] for r in test_references]
    return {
        "bleu": bleu.compute(predictions=predictions, references=refs_bleu)["score"],
        "meteor": compute_meteor(predictions, test_references),
        "chrf": chrf.compute(predictions=predictions, references=refs_bleu)["score"],
    }


def evaluate_marian_pipeline(test_sources, test_references):
    translator = pipeline("translation_en_to_de", model=MODEL_NAME, device=0)
    results = translator(test_sources, max_length=MAX_LENGTH, batch_size=BATCH_SIZE)
    predictions = [r["translation_text"] for r in results]
    refs_bleu = [[r] for r in test_references]
    return {
        "bleu": bleu.compute(predictions=predictions, references=refs_bleu)["score"],
        "meteor": compute_meteor(predictions, test_references),
        "chrf": chrf.compute(predictions=predictions, references=refs_bleu)["score"],
    }