from . import Tokenizer
from typing import List, Tuple, Union
import os
import json
import torch
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from keras_preprocessing.sequence import pad_sequences
import numpy as np


def build_tokenizer(data: List[str]) -> Tokenizer:
    """
    Build a tokenizer by assembling its vocabulary and word <-> index mappings.
    """
    tokenizer = Tokenizer()
    for sentence in data:
        tokenizer.process_sentence(sentence)

    tokenizer.seal_tokenizer()

    return tokenizer


def tokenize_data(data: List[str], tokenizer: Tokenizer) -> List[List[int]]:
    """
    Return a list of integers corresponding to the tokenizer's indices of each
    word in each sentence given in <data>.
    """
    tokenized_data = []
    for sentence in data:
        tokenized_data.append(tokenizer.encode(sentence))

    return tokenized_data


def pad_data(data: List[List[int]], batch_size: int, tokenizer: Tokenizer,
             return_longest_seq_len=True) -> Union[List[List[int]],
                                                   Tuple[List[List[int]], int]]:
    """
    Pad the data by dividing the entire dataset into batches of size
    <batch_size> and then for each batch, post-pad the data with the tokenizer's
    padding token. For each sentence in a batch, the difference in length
    between this sentence and the longest sentence in the batch will be padded.
    """
    longest_seq_len = 0
    padded_data = []
    for i in range(start=0, stop=len(data), step=batch_size):
        batch = data[i: i + batch_size]
        max_len = max([len(sentence) for sentence in batch])
        if max_len > longest_seq_len:
            longest_seq_len = max_len

        padded_data.append(pad_sequences(
            batch, maxlen=max_len, dtype="long", value=tokenizer.pad_token,
            truncating="post", padding="post"
        ))

    if return_longest_seq_len:
        return padded_data, longest_seq_len
    else:
        return padded_data



def process_data(path: str, batch_size: int, return_misc=True) \
        -> Union[Tuple[DataLoader, DataLoader],
                 Tuple[DataLoader, DataLoader, Tokenizer, int]]:
    """
    Return 2 torch data loaders from the data path given for training the
    chatbot, one for training and one for validation.

    <path> should be a string to a directory within the current file system
    storing all the JSON files containing the conversational data formatted
    in the following manner:
    [
        {
            "input": {
                "sender_name": ...,
                "message": ...
            },
            "target": {
                "sender_name": ...,
                "message": ...
            }
        },
        ...
    ]

    <batch_size> is used to split the data into batches of size <batch_size> for
    batch gradient descent.
    """
    inputs = []
    targets = []
    for file_name in os.listdir(path):
        with open(file_name) as data_file:
            data = json.load(data_file)
            for message_pair in data:
                inputs.append(message_pair["input"]["message"])
                targets.append(message_pair["target"]["message"])

    # here we build the tokenizer using both the inputs and targets because
    # we want the tokenizer to have access to all of words in the dataset
    tokenizer = build_tokenizer(inputs + targets)

    tokenized_inputs = tokenize_data(inputs, tokenizer)
    tokenized_targets = tokenize_data(targets, tokenizer)

    padded_inputs, longest_input_seq = pad_data(
        tokenized_inputs, batch_size, tokenizer)
    padded_targets, longest_output_seq = pad_data(
        tokenized_targets, batch_size, tokenizer)

    longest_seq_len = max(longest_input_seq, longest_output_seq)

    inputs = torch.tensor(padded_inputs)
    targets = torch.tensor(padded_targets)

    # we will train using 90% of our dataset and use the other 10% for
    # validation
    permuted_indices = torch.randperm(inputs.size(0))
    num_train = np.ceil(0.9 * inputs.size(0))

    train_indices = permuted_indices[: num_train]
    val_indices = permuted_indices[num_train: ]

    train_inputs, train_targets = inputs[train_indices], targets[train_indices]
    val_inputs, val_targets = inputs[val_indices], targets[val_indices]

    train_data = TensorDataset(train_inputs, train_targets)
    train_sampler = SequentialSampler(train_data)
    train_dataloader = DataLoader(train_data, batch_size, sampler=train_sampler)

    val_data = TensorDataset(val_inputs, val_targets)
    val_sampler = SequentialSampler(val_data)
    val_dataloader = DataLoader(val_data, batch_size, sampler=val_sampler)

    if return_misc:
        return train_dataloader, val_dataloader, tokenizer, longest_seq_len
    else:
        return train_dataloader, val_dataloader
