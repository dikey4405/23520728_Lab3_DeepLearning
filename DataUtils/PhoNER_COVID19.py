import torch
from torch import load, nn
import torch.nn.utils.rnn as pad_sequence
import torch.nn.functional as F
from torch.utils.data import Dataset

import json
import os
import string
from collections import Counter

class Vocab:
    def __init__(self, path: list[str], min_freq: int):
        word_counter = Counter()
        tag_set = set()

        for filename in os.listdir(path):
            data = json.load(open(os.path.join(path, filename)))
            for item in data:
                tokens = item["tokens"]
                tags = item["tags"]

                for toks in tokens:
                    word_counter[toks.lower()] += 1
                for tag in tags:
                    tag_set.add(tag)
        
        self.pad = "<pad>"
        self.unk = "<unk>"
        self.pad_tag = "<pad_tag>"

        self.word2idx = {
            self.pad: 0,
            self.unk: 1,
        }

        for w, c in word_counter.items():
            if c >= min_freq:
                self.word2idx[w] = len(self.word2idx)
        
        self.idx2word = {i: w for w, i in self.word2idx.items()}

        self.tag2idx = {
            tag: i for i, tag in enumerate(sorted(list(tag_set)))
        }
        self.tag2idx[self.pad_tag] = -100
        self.idx2tag = {i: t for t, i in self.tag2idx.items()}
    
    def encode_tokens(self, tokens: list[str]) -> torch.Tensor:
        ids = []
        for token in tokens:
            token = token.lower()
            ids.append(self.word2idx.get(token, self.word2idx[self.unk]))
        return torch.tensor(ids, dtype=torch.long)
    
    def encode_tags(self, tags: list[str]) -> torch.Tensor:
        ids = [self.tag2idx[tag] for tag in tags]
        return torch.tensor(ids, dtype=torch.long)
    
    @property
    def n_tags(self) -> int:
        return len([i for i in self.tag2idx.values() if i != -100])
    
    @property
    def vocab_size(self) -> int:
        return len(self.word2idx)
    
class PhoNER_COVID19(Dataset):
    def __init__(self, path: str, vocab: Vocab):
        super().__init__()
        self.vocab = vocab
        self.data = json.load(open(path))
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, index: int) -> dict:
        item = self.data[index]

        tokens: list[str] = item["tokens"]
        tags: list[str] = item["tags"]

        input_ids = self.vocab.encode_tokens(tokens)
        tags_ids = self.vocab.encode_tags(tags)

        return {
            "input_ids": input_ids,
            "tags_ids": tags_ids
        }
def collate_fn(items: dict) -> torch.Tensor:
    input_ids = [item["input_ids"] for item in items]
    tags_ids = [item["tags_ids"] for item in items]
    
    lengths = torch.tensor([len(seq) for seq in input_ids], dtype=torch.long)
    
    padded_input = pad_sequence(input_ids, batch_first=True, padding_value=0)
    padded_tags = pad_sequence(tags_ids, batch_first=True, padding_value=-100)

    return {
        "input_ids": padded_input,
        "tags_ids": padded_tags,
        "lengths": lengths
    }