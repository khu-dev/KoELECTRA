import os

import torch
import numpy as np
from flask import Flask, request, jsonify
from transformers import (ElectraTokenizer, ElectraForSequenceClassification)

app = Flask(__name__)
device = "cuda" if torch.cuda.is_available() else "cpu"

max_seq_length = int(os.getenv("PODOLI_MAX_LENGTH", 128))
model = ElectraForSequenceClassification.from_pretrained('model')
model.to(device)

tokenizer = ElectraTokenizer.from_pretrained(
    "monologg/koelectra-small-v3-discriminator",
    do_lower_case=False
)

def featurize(comments):
    tokens_a = tokenizer.tokenize(comments)
    if len(tokens_a) > max_seq_length - 2:
        tokens_a = tokens_a[0:(max_seq_length - 2)]

    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    input_mask = [1] * len(input_ids)

    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    return (input_ids, input_mask)

def get_malicious_score(input_ids, attention):
    attention = attention.cpu().tolist()
    malicious_score = []
    for i, (input_id, attention_value) in enumerate(zip(input_ids, attention)):
        token = tokenizer.convert_ids_to_tokens(input_id)
        if token not in list(tokenizer.special_tokens_map.values()):
            malicious_score.append((token, attention_value))
    return malicious_score

@app.route('/', methods=['GET'])
def ping():
    return 'pong'

@app.route('/analyzer', methods=['POST'])
def analyzer():
    comment = request.json['comment']
    (input_ids, input_mask) = featurize(comment)

    model.eval()
    all_input_ids = torch.tensor([input_ids], dtype=torch.long).to(device)
    all_attention_mask = torch.tensor([input_mask], dtype=torch.long).to(device)
    with torch.no_grad():
        inputs = {
            "input_ids": all_input_ids,
            "attention_mask": all_attention_mask,
        }
        outputs = model(**inputs, output_attentions=True, return_dict=True)
        logits = outputs.logits
        last_attention_cls = outputs.attentions[-1][0, -1, 0]
        preds = np.argmax(logits.detach().cpu().numpy(), axis=1)
        malicious_score = get_malicious_score(input_ids=input_ids, attention=last_attention_cls)

        malicious = True if preds[0] == 0 else False
        print(malicious_score)
        return jsonify(malicious=malicious, malicious_score=malicious_score)

if __name__ == '__main__':
    app.run(host= '0.0.0.0', debug=True, port=int(os.getenv("PORT", "8000")))