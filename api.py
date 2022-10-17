from fastapi import FastAPI, Request
from datetime import datetime
from functools import wraps
from http import HTTPStatus
from pathlib import Path
from typing import Dict, List
import numpy as np
import pandas as pd
import random
import torch
import torch.nn.functional as F
import shutil
import os
import pickle


if torch.cuda.is_available():
    device='gpu'
else:
    device=torch.device('cpu')

MODELS_DIR = Path("models/")
DATA_DIR = Path("data_files/")


class Dictionary(object):
    def __init__(self):
        self.token2idx = {}
        self.idx2token = []

    def add_token(self, token):
        if token not in self.token2idx:
            self.idx2token.append(token)
            self.token2idx[token] = len(self.idx2token) - 1
        return self.token2idx[token]

    def __len__(self):
        return len(self.idx2token)


class CharRNNClassifier(torch.nn.Module):
    
    def __init__(self, input_size, embedding_size, hidden_size, output_size, model="lstm", num_layers=2,
                 bidirectional=False, pad_idx=0):
        super().__init__()
        self.model = model.lower()
        self.hidden_size = hidden_size
        self.embed = torch.nn.Embedding(input_size, embedding_size, padding_idx=pad_idx)
        if self.model == "gru":
            self.rnn = torch.nn.GRU(embedding_size, hidden_size, num_layers, bidirectional=bidirectional)
        elif self.model == "lstm":
            self.rnn = torch.nn.LSTM(embedding_size, hidden_size, num_layers, bidirectional=bidirectional)
        self.h2o = torch.nn.Linear(hidden_size, output_size)
        
    def forward(self, input, input_lengths):
        encoded = self.embed(input)
        packed = torch.nn.utils.rnn.pack_padded_sequence(encoded, input_lengths)
        output, _ = self.rnn(packed)
        padded_mean, _ = torch.nn.utils.rnn.pad_packed_sequence(output, padding_value=0.0)
        padded_max, _ = torch.nn.utils.rnn.pad_packed_sequence(output, padding_value=float('-inf'))
        max_layer, _ = padded_max.max(dim=0)
        mean_layer = padded_mean.mean(dim=0)
        output = max_layer + mean_layer
        output = self.h2o(output)
        return output


def batch_generator(data, batch_size, token_size):
    minibatch, sequences_so_far, tokens_so_far = [], 0, 0
    for ex in data:
        seq_len = len(ex[0])
        if seq_len > token_size:
            ex = (ex[0][:token_size], ex[1])
            seq_len = token_size
        minibatch.append(ex)
        sequences_so_far += 1
        tokens_so_far += seq_len
        if sequences_so_far == batch_size or tokens_so_far == token_size:
            yield minibatch
            minibatch, sequences_so_far, tokens_so_far = [], 0, 0
        elif sequences_so_far > batch_size or tokens_so_far > token_size:
            yield minibatch[:-1]
            minibatch, sequences_so_far, tokens_so_far = minibatch[-1:], 1, len(minibatch[-1][0])
    if minibatch:
        yield minibatch


def pool_generator(data, batch_size, token_size, shuffle=False):
    for p in batch_generator(data, batch_size * 100, token_size * 100):
        p_batch = batch_generator(sorted(p, key=lambda t: len(t[0]), reverse=True), batch_size, token_size)
        p_list = list(p_batch)
        if shuffle:
            for b in random.sample(p_list, len(p_list)):
                yield b
        else:
            for b in p_list:
                yield b


def test(model, data, batch_size, token_size):
    model.eval()
    sindex = []
    labels = []
    with torch.no_grad():
        for batch in pool_generator(data, batch_size, token_size):
            # Get input sequences from batch
            X = [torch.from_numpy(d[0]) for d in batch]
            X_lengths = torch.tensor([x.numel() for x in X], dtype=torch.long)
            # Pad the input sequences to create a matrix
            X = torch.nn.utils.rnn.pad_sequence(X).to(device)
            answer = model(X, X_lengths)
            label = torch.max(answer, 1)[1].cpu().numpy()
            # Save labels and sentences index
            labels.append(label)
            sindex += [d[1] for d in batch]
            
    index = np.array(sindex)
    labels = np.concatenate(labels)
    order = np.argsort(index)
    labels = labels[order]
    labels = [l+1 for l in labels]
    
    return labels


# Initializing the API
app = FastAPI(
    title="SentText - Amazon reviews predictor",
    description="This API lets you introduce the text of an Amazon review and predicts whether it is positive (class 2) or negative (class 1).",
    version="0.1",
)


def construct_response(f):
    """Construct a JSON response for an endpoint's results."""

    @wraps(f)
    def wrap(request: Request, *args, **kwargs):
        results = f(request, *args, **kwargs)
        
        response = {
            "message": results["message"],
            "method": request.method,
            "status_code": results["status_code"],
            "timestamp": datetime.now().isoformat(),
            "url": request.url._url,
        }
        if "data" in results:
            response["data"] = results["data"]
        
        return response

    return wrap


@app.on_event("startup")
def _load_model_vocabulary():
    #Loads the classifier model
    model_paths = [
        filename for filename in MODELS_DIR.iterdir() if filename.suffix == ".pt"
    ]
    data_paths = [
        filename for filename in DATA_DIR.iterdir() if filename.suffix == ".csv"
    ]

    char_vocab = Dictionary()
    pad_token = '<pad>'
    unk_token = '<unk>'
    pad_index = char_vocab.add_token(pad_token)
    unk_index = char_vocab.add_token(unk_token)

    data0 = pd.read_csv(data_paths[0])
    data1 = pd.read_csv(data_paths[1])
    chars_train = set(''.join(data1['review_text']))
    chars_test = set(''.join(data0['review_text']))
    chars = chars_train.union(chars_test)
            
    for char in sorted(chars):
        char_vocab.add_token(char)
            
    class_vocab = Dictionary()
    classes = set(data0['class'])
    for c in sorted(classes):
        class_vocab.add_token(c)

    input_size = len(char_vocab)
    embedding_size = 32
    hidden_size = 256
    output_size = len(class_vocab)

    model = CharRNNClassifier(input_size, embedding_size, hidden_size, output_size)
    model.load_state_dict(torch.load(model_paths[0], map_location = device))
    model.eval()
    return char_vocab, class_vocab, model


CHAR_VOCAB, CLASS_VOCAB, MODEL = _load_model_vocabulary()


@app.get("/", tags=["General"])
@construct_response
def _index(request: Request):

    response = {
        "message": HTTPStatus.OK.phrase,
        "status_code": HTTPStatus.OK,
        "data": {"message": "Welcome to SentText Amazon reviews classifier! Please, read the /docs and enter a review in /Prediction section!"},
    }
    return response


def Prediction(input_review):
    x_test_idx = [np.array([CHAR_VOCAB.token2idx[c] for c in input_review])]
    test_data = [(x, idx) for idx, x in enumerate(x_test_idx)]

    # Predict the output
    batch_size = 32
    token_size = 100000
    device = 'cpu'
    labels = test(MODEL, test_data, batch_size, token_size)
    return labels


def Prediction_name(labels):
    if labels == 2:
        return 'It is a positive review.'
    else:
        return 'It is a negative review.'


@app.post("/", tags=["Prediction"])
@construct_response
def _predict(request: Request, review: str):
    try:
        labels = int(Prediction(review)[0])
        prediction_type = str(Prediction_name(labels))
        response = {
            "message": HTTPStatus.OK.phrase,
            "status_code": HTTPStatus.OK,
            "data": {
                "Prediction": labels,
                "Prediction type": prediction_type
            }
        }
    except:
        response = {
            "message": HTTPStatus.BAD_REQUEST.phrase,
            "status_code": HTTPStatus.BAD_REQUEST
        }
    return response
    