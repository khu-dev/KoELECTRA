import torch
import random
import argparse
from tqdm import tqdm
import torch.autograd.profiler as profiler
from transformers import (ElectraTokenizer, ElectraForSequenceClassification)

device = "cuda" if torch.cuda.is_available() else "cpu"

model = ElectraForSequenceClassification.from_pretrained('model')
model.to(device)

tokenizer = ElectraTokenizer.from_pretrained(
    "monologg/koelectra-small-v3-discriminator",
    do_lower_case=False
)

def analyzer():
    matrix = torch.randint(0, len(tokenizer) - 1, (args.batch, args.max_seq_length))
    all_input_ids = torch.tensor(matrix, dtype=torch.long).to(device)
    all_attention_mask = torch.tensor(matrix, dtype=torch.long).to(device)
    inputs = {
        "input_ids": all_input_ids,
        "attention_mask": all_attention_mask,
    }
    with profiler.profile(profile_memory=True, record_shapes=True) as prof:
        model(**inputs)

    total_average = prof.total_average()

    return total_average.cpu_memory_usage

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, required=True)
    parser.add_argument('--max_seq_length', type=int, required=True)
    args = parser.parse_args()

    result = []
    for i in tqdm(range(10)):
        result.append(analyzer())
    print(sum(result) / len(result))

    # cpu_memory_usage in t2.small (2GB)
    # batch size 4
    # 64 :  700125328 Byte
    # 128 : 1588957328 Byte
    # 256 : 3932852368 Byte
    # 512 : Dead