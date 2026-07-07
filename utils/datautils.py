# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import json
import logging
import random
from dataclasses import dataclass
from typing import Dict, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets import load_dataset
from transformers import AutoTokenizer


IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

def get_loaders(tokenizer, name, seed=0, seqlen=2048, cache_dir=None):
    if "wikitext2" in name:
        testdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="test", cache_dir=cache_dir)
        return tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")
    elif "c4" in name:
        valdata = load_dataset(
            "allenai/c4",
            data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
            split="validation",
            cache_dir=cache_dir,
        )
        random.seed(seed)
        valenc = []
        for _ in range(256):
            while True:
                i = random.randint(0, len(valdata) - 1)
                tmp = tokenizer(valdata[i]["text"], return_tensors="pt")
                if tmp.input_ids.shape[1] > seqlen:
                    break
            i = random.randint(0, tmp.input_ids.shape[1] - seqlen - 1)
            valenc.append(tmp.input_ids[:, i : i + seqlen])
        return torch.hstack(valenc)
    raise NotImplementedError(f"Unsupported PPL dataset: {name}")
    
def get_wikitext2(tokenizer: AutoTokenizer,  sequence_length: int):
    test_dataset_raw = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    test_dataset_tok = tokenizer("\n\n".join(test_dataset_raw["text"]), return_tensors="pt").input_ids
    num_test_sequences = test_dataset_tok.numel() // sequence_length
    test_loader = []
    for i in range(num_test_sequences):
        test_loader.append(test_dataset_tok[:, i * sequence_length : (i + 1) * sequence_length])
    return test_loader


def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)


def get_train_val_dataset(train_path, valid_path=None):
    f = open(train_path, "r", encoding="utf-8")
    data = []
    while True:
        line = f.readline()
        if not line:
            break
        data.append(json.loads(line))
    f.close()
    train_data = []
    valid_data = []
    if valid_path:
        f = open(valid_path, "r", encoding="utf-8")
        while True:
            line = f.readline()
            if not line:
                break
            valid_data.append(json.loads(line))
        f.close()
        train_data = data
    else:
        train_data = data[10000:]
        valid_data = data[:10000]
    return train_data, valid_data


'''
全量加载占用内存太大，以及有冗余（不需要 input_ids 以外的键）；缺少 eos 
class CustomJsonDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, tokenizer, block_size=1024):
        raw_data = dataset
        self.tokenizer = tokenizer
        self.block_size = block_size
        tokenized_datasets = []
        for d in raw_data:
            tokenized_datasets.append(self.tokenize_function(d))

        grouped_dataset = self.group_texts(tokenized_datasets)
        self.input_ids = grouped_dataset["input_ids"]
        self.labels = grouped_dataset["labels"]
        self.data = [
            dict(input_ids=self.input_ids[i], labels=self.labels[i])
            for i in range(len(self.input_ids))
        ]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i):
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])

    def __iter__(self):
        return iter(self.data)

    def tokenize_function(self, examples):
        return self.tokenizer(examples["text"])

    def group_texts(self, examples):
        # Concatenate all texts.
        # Initialize an empty dictionary
        concatenated_examples = {}

        # Loop through the list of dictionaries
        for d in examples:
            # Loop through the keys in each dictionary
            for key in d.keys():
                # If the key is not already a key in the dict_of_lists, create a new list
                if key not in concatenated_examples:
                    concatenated_examples[key] = []
                # Append the value to the list associated with the key in dict_of_lists
                concatenated_examples[key].extend(d[key])
        total_length = len(concatenated_examples["input_ids"])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= self.block_size:
            total_length = (total_length // self.block_size) * self.block_size
        # Split by chunks of max_len.
        result = {
            k: [
                t[i : i + self.block_size]
                for i in range(0, total_length, self.block_size)
            ]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result
'''

class CustomBinDataset(torch.utils.data.Dataset):
    def __init__(self, bin_path, block_size=1024):
        self.block_size = block_size
        # Memory map the binary file (read-only mode)
        self.data = np.memmap(bin_path, dtype=np.uint32, mode='r')
        self.num_blocks = len(self.data) // self.block_size

    def __len__(self):
        return self.num_blocks

    def __getitem__(self, idx):
        start = idx * self.block_size
        end = start + self.block_size
        
        # Load the slice into memory and cast to LongTensor (int64)
        input_ids = torch.from_numpy(self.data[start:end].astype(np.int64))
        labels = input_ids.clone()
        return dict(input_ids=input_ids, labels=labels)


def preprocess_jsonl_to_bin_split(train_path, valid_path, cache_dir, tokenizer, max_train_tokens=None):
    """
    Streams JSONL data to binary format using uint32.
    If valid_path is None, splits the first 10,000 lines of train_path into validation.
    """
    import os
    import torch.distributed as dist
    
    os.makedirs(cache_dir, exist_ok=True)
    train_bin = os.path.join(cache_dir, "train.bin")
    valid_bin = os.path.join(cache_dir, "val.bin")

    if not dist.is_initialized() or dist.get_rank() == 0:
        # If already preprocessed, skip
        if os.path.exists(train_bin) and os.path.exists(valid_bin):
            print(f"Preprocessed binary cache files found at {cache_dir}. Skipping preprocessing.")
        else:
            print(f"Starting offline preprocessing of datasets into binary format...")
            print(f"Cache directory: {cache_dir}")

            # Helper function to stream a jsonl iterator into a binary file
            def stream_to_bin(line_iterator, bin_filepath, max_tokens=None):
                chunk_tokens = []
                chunk_limit = 1000000
                total_tokens = 0
                line_count = 0
                batch_texts = []
                batch_size = 16384

                def process_batch(texts):
                    # Tokenize batch of texts in parallel using Rust threads
                    batch_outputs = tokenizer(texts, add_special_tokens=False)["input_ids"]
                    batch_ids = []
                    for ids in batch_outputs:
                        ids.append(tokenizer.eos_token_id)
                        batch_ids.extend(ids)
                    return batch_ids

                with open(bin_filepath, "wb") as f_out:
                    for line in line_iterator:
                        if not line.strip():
                            continue
                        item = json.loads(line)
                        batch_texts.append(item["text"])
                        line_count += 1

                        if len(batch_texts) >= batch_size:
                            ids = process_batch(batch_texts)
                            
                            # Check if adding this batch exceeds max_tokens limit
                            if max_tokens is not None and total_tokens + len(chunk_tokens) + len(ids) >= max_tokens:
                                needed = max_tokens - (total_tokens + len(chunk_tokens))
                                ids = ids[:needed]
                                chunk_tokens.extend(ids)
                                
                                # Write the final chunk and exit early
                                arr = np.array(chunk_tokens, dtype=np.uint32)
                                f_out.write(arr.tobytes())
                                total_tokens += len(chunk_tokens)
                                chunk_tokens = []
                                batch_texts = []
                                break
                            
                            chunk_tokens.extend(ids)
                            batch_texts = []

                        if line_count % 50000 == 0:
                            print(f"  Processed {line_count} lines...")

                        if len(chunk_tokens) >= chunk_limit:
                            arr = np.array(chunk_tokens, dtype=np.uint32)
                            f_out.write(arr.tobytes())
                            total_tokens += len(chunk_tokens)
                            chunk_tokens = []
                    
                    # Process remaining batch (only if we didn't break early)
                    if batch_texts:
                        ids = process_batch(batch_texts)
                        if max_tokens is not None and total_tokens + len(chunk_tokens) + len(ids) >= max_tokens:
                            needed = max_tokens - (total_tokens + len(chunk_tokens))
                            ids = ids[:needed]
                        chunk_tokens.extend(ids)

                    # Write remaining chunk
                    if chunk_tokens:
                        arr = np.array(chunk_tokens, dtype=np.uint32)
                        f_out.write(arr.tobytes())
                        total_tokens += len(chunk_tokens)
                print(f"Saved {bin_filepath} with {total_tokens} tokens (processed {line_count} lines).")

            # Generator for files
            def line_generator(filepath, start_line=0, end_line=None):
                with open(filepath, "r", encoding="utf-8") as f:
                    for idx, line in enumerate(f):
                        if idx < start_line:
                            continue
                        if end_line is not None and idx >= end_line:
                            break
                        yield line

            if valid_path:
                print("Preprocessing validation dataset...")
                stream_to_bin(line_generator(valid_path), valid_bin)
                print("Preprocessing training dataset...")
                stream_to_bin(line_generator(train_path), train_bin, max_tokens=max_train_tokens)
            else:
                # Original logic: train_data = data[10000:], valid_data = data[:10000]
                print("Splitting train dataset: first 10,000 lines as validation dataset...")
                stream_to_bin(line_generator(train_path, end_line=10000), valid_bin)
                print("Splitting train dataset: remaining lines as training dataset...")
                stream_to_bin(line_generator(train_path, start_line=10000), train_bin, max_tokens=max_train_tokens)

    if dist.is_initialized():
        dist.barrier()

    return train_bin, valid_bin


def jload(filename, mode="r"):
    """Load a .json file into a dictionary."""
    with open(filename, mode) as f:
        jdict = json.load(f)
    return jdict

