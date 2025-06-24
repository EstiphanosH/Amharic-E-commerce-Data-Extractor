from datasets import Dataset, ClassLabel
from transformers import AutoTokenizer
import pandas as pd

def load_conll_data(file_path):
    """Load CoNLL formatted data into Hugging Face Dataset"""
    tokens, labels = [], []
    current_tokens, current_labels = [], []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                if current_tokens:
                    tokens.append(current_tokens)
                    labels.append(current_labels)
                    current_tokens, current_labels = [], []
            else:
                parts = line.split()
                current_tokens.append(parts[0])
                current_labels.append(parts[1])
    
    return Dataset.from_dict({
        'tokens': tokens,
        'ner_tags': labels
    })

def tokenize_and_align_labels(dataset, tokenizer, label2id):
    """Tokenize text and align NER labels with subword tokens"""
    tokenized_inputs = tokenizer(
        dataset["tokens"],
        truncation=True,
        is_split_into_words=True,
        padding="max_length",
        max_length=128
    )
    
    labels = []
    for i, label in enumerate(dataset["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label2id.get(label[word_idx], -100))
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
            
        labels.append(label_ids)
    
    tokenized_inputs["labels"] = labels
    return tokenized_inputs