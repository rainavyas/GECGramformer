'''
Generate model predictions using the Gramformer

Input file:
ID1 Sentence1
ID2 Sentence2
.
.
.

Output file:
ID1 Sentence1
ID2 Sentence2
.
.
.
'''

import sys
import os
import argparse
import torch
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM

def get_sentences(data_path):
    with open(data_path, 'r') as f:
        lines = f.readlines()
    texts = [' '.join(l.rstrip('\n').split()[1:]) for l in lines]
    ids = [l.rstrip('\n').split()[0] for l in lines]
    return ids, texts

def correct(model, tokenizer, sentence):
    correction_prefix = "gec: "
    sentence = correction_prefix + sentence
    input_ids = tokenizer.encode(sentence, return_tensors='pt')
    prediction_ids = model.generate(
        input_ids,
        do_sample=True, 
        max_length=128, 
        top_k=50, 
        top_p=0.95, 
        early_stopping=True,
        num_return_sequences=1)
    return tokenizer.decode(prediction_ids.squeeze(), skip_special_tokens=True, clean_up_tokenization_spaces=True)

if __name__ == "__main__":

    # Get command line arguments
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('IN', type=str, help='Path to input data')
    commandLineParser.add_argument('OUT', type=str, help='Path to corrected output data')
    commandLineParser.add_argument('--seed', type=int, default=1, help='Seed for reproducibility')
    args = commandLineParser.parse_args()

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/predict.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n') 
    torch.manual_seed(args.seed)
    
    # Load Model and Tokenizer
    correction_model_tag = "prithivida/grammar_error_correcter_v1"
    tokenizer = AutoTokenizer.from_pretrained(correction_model_tag)
    model = AutoModelForSeq2SeqLM.from_pretrained(correction_model_tag)

    # Load input sentences
    identifiers, sentences = get_sentences(args.IN)

    # Correction (prediction) for each input sentence
    corrections = []
    for i, sent in enumerate(sentences):
        print(f'On {i}/{len(sentences)}')
        corrections.append(correct(model, tokenizer, sent))
    assert len(corrections) == len(identifiers), "Number of ids don't match number of predictions"

    # Save predictions
    with open(args.OUT, 'w') as f:
        for id, sentence in zip(identifiers, corrections):
            f.write(f'{id} {sentence}\n')

