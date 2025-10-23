# Code adapted from https://github.com/IST-DASLab/sparsegpt/blob/master/datautils.py

import numpy as np
import random
import torch
from datasets import load_dataset
from datasets import load_from_disk


# Set random seed for reproducibility
def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)

# Wrapper class for tokenized input IDs
class TokenizerWrapper:
    """
    Wrapper class for tokenized input IDs.

    Args:
        input_ids (tensor): The tokenized input IDs from the tokenizer.
    """
    def __init__(self, input_ids):
        self.input_ids = input_ids

# Load and process wikitext2 dataset from local
def get_wikitext2_local(nsamples, seed, seqlen, tokenizer):
    """
    Load and process the Wikitext-2 dataset.

    Args:
        nsamples (int): Number of samples to generate from the training set.
        seed (int): Random seed for reproducibility.
        seqlen (int): Sequence length for generated samples.
        tokenizer (Tokenizer): Tokenizer instance for encoding texts.

    Returns:
        tuple: A tuple containing trainloader (list of input and target pairs) and encoded test dataset.
    """
    # Load train and test datasets
    all_data = load_from_disk('./data_local/wiki_all')
    traindata = all_data['train']
    testdata = all_data['test']

    # Encode datasets
    trainenc = tokenizer(" ".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    # Generate samples from training set using random seed and specified sequence length
    random.seed(seed)
    trainloader = []
    print(type(nsamples))
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    return trainloader, testenc


# Load and process wikitext2 dataset
def get_wikitext2(nsamples, seed, seqlen, tokenizer):
    """
    Load and process the Wikitext-2 dataset.

    Args:
        nsamples (int): Number of samples to generate from the training set.
        seed (int): Random seed for reproducibility.
        seqlen (int): Sequence length for generated samples.
        tokenizer (Tokenizer): Tokenizer instance for encoding texts.

    Returns:
        tuple: A tuple containing trainloader (list of input and target pairs) and encoded test dataset.
    """
    # Load train and test datasets
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')

    # Encode datasets
    # trainenc = tokenizer(" ".join(traindata['text']), return_tensors='pt')
    trainenc_list = []
    chunk_size = seqlen * 100
    
    for i in range(0, len(traindata['text']), chunk_size):
        chunk = " ".join(traindata['text'][i:i + chunk_size])
        enc = tokenizer(chunk, return_tensors='pt')
        trainenc_list.append(enc.input_ids)
    
    trainenc = torch.cat(trainenc_list, dim=1)
    print("trainenc_complete")
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')
    print("testenc_complete")

    # Generate samples from training set using random seed and specified sequence length
    random.seed(seed)
    trainloader = []
    print(type(nsamples))
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc

# Function to select the appropriate loader based on dataset name
def get_loaders(nsamples=128, seed=0, seqlen=2048, tokenizer=None):
    return get_wikitext2_local(nsamples, seed, seqlen, tokenizer)

if __name__ == "__main__":
    get_loaders('wikitext2', seed=0, seqlen=2048, tokenizer=None)

