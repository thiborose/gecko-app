import argparse

from application.models.gector.utils.helpers import read_lines
from application.models.gector.gector.gec_model import GecBERTModel


def predict_for_string(input_text, model, batch_size=32):
    predictions = []
    cnt_corrections = 0
    batch = []
    for sent in input_text.split('.'): # segmentation with spacy
        batch.append(sent.split())
        if len(batch) == batch_size:
            preds, cnt = model.handle_batch(batch)
            predictions.extend(preds)
            cnt_corrections += cnt
            batch = []
    if batch:
        preds, cnt = model.handle_batch(batch)
        predictions.extend(preds)
        cnt_corrections += cnt

    output_text = " ".join([" ".join(x) for x in predictions])
    return output_text


def predict_for_tokens(input_text, model, batch_size=32):
    predictions = []
    cnt_corrections = 0
    batch = []
    for sent in input_text.split('.'): # segmentation with spacy
        batch.append(sent)
        if len(batch) == batch_size:
            preds, cnt = model.handle_batch(batch)
            predictions.extend(preds)
            cnt_corrections += cnt
            batch = []
    if batch:
        preds, cnt = model.handle_batch(batch)
        predictions.extend(preds)
        cnt_corrections += cnt

    output_tokens = predictions
    return output_tokens



def predict_for_file(input_file, output_file, model, batch_size=32):
    test_data = read_lines(input_file)
    predictions = []
    cnt_corrections = 0
    batch = []
    for sent in test_data:
        batch.append(sent.split())
        if len(batch) == batch_size:
            preds, cnt = model.handle_batch(batch)
            predictions.extend(preds)
            cnt_corrections += cnt
            batch = []
    if batch:
        preds, cnt = model.handle_batch(batch)
        predictions.extend(preds)
        cnt_corrections += cnt

    with open(output_file, 'w') as f:
        f.write("\n".join([" ".join(x) for x in predictions]) + '\n')
    return cnt_corrections
