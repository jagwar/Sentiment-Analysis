import os
import json
import numpy as np

import torch
from torch.utils.data import DataLoader, RandomSampler, TensorDataset, SequentialSampler

from transformers import CamembertTokenizer, CamembertForSequenceClassification
import pandas as pd
from tqdm import tqdm, trange


# tokenizer = CamembertTokenizer.from_pretrained('/home/crannou/workspace/sentiment-eai/data/36e8f471-821d-4270-be56-febb1be36c26')
# model = CamembertForSequenceClassification.from_pretrained('/home/crannou/workspace/sentiment-eai/data/36e8f471-821d-4270-be56-febb1be36c26')

# tokenizer = CamembertTokenizer.from_pretrained('/home/crannou/workspace/sentiment-eai/7a37b1e5-8e7b-45d1-9e87-7314e8e66c0c/')
# model = CamembertForSequenceClassification.from_pretrained('/home/crannou/workspace/sentiment-eai/7a37b1e5-8e7b-45d1-9e87-7314e8e66c0c/')

tokenizer = CamembertTokenizer.from_pretrained('/home/crannou/workspace/serving-preset-images/sentiment-analysis-fr/app/model_sources')
model = CamembertForSequenceClassification.from_pretrained('/home/crannou/workspace/serving-preset-images/sentiment-analysis-fr/app/model_sources')

def eval_model():
    df = pd.read_csv('/home/crannou/notebooks/review_polarity_bin.csv', sep=';')
    preds = []
    all_input_ids = []
    all_attention_masks = []
    df = df.sample(frac=0.1, random_state=42)
    all_labels = df['polarity'].values
    for sentence in df['review_content']:
        input_ids, attention_mask = get_features(sentence)
        all_input_ids.append(input_ids)
        all_attention_masks.append(attention_mask)

    t_inputs_ids = torch.tensor(all_input_ids, dtype=torch.long)
    t_attention_mask = torch.tensor(all_attention_masks, dtype=torch.long)
    t_labels = torch.tensor(all_labels, dtype=torch.long)
    dataset = TensorDataset(t_inputs_ids, t_attention_mask, t_labels)

    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(
        dataset, sampler=eval_sampler, batch_size=32
    )

    model.eval()
    preds = None
    out_label_ids = None
    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            batch = tuple(t.to("cpu") for t in batch)

            inputs = {
                "input_ids": batch[0], "attention_mask": batch[1], "labels": batch[2]}

            outputs = model(**inputs)
            _, logits = outputs[:2]

            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs["labels"].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(
                    out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

    preds = np.argmax(preds, axis=1)

    result = {"acc": (preds == out_label_ids).mean()}
    print(result)

def get_features(sentence):
    max_length=min(128, tokenizer.max_len)
    input_ids = tokenizer.encode(
        sentence, add_special_tokens=True, max_length=min(128, tokenizer.max_len),
    )
    padding_length = max_length - len(input_ids)
    attention_mask = [1] * len(input_ids)
    input_ids = input_ids + ([0] * padding_length)
    attention_mask = attention_mask + ([0] * padding_length)
    return input_ids, attention_mask


if __name__ == '__main__':
    eval_model()