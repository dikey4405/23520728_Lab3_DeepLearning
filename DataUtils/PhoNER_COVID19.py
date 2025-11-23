import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import json
import os
from collections import Counter

class Vocab:
    def __init__(self, train_filepath: str, min_freq: int = 1):
        word_counter = Counter()
        tag_set = set()

        print(f"Building vocab from: {train_filepath}")
        try:
            data = json.load(open(train_filepath, encoding='utf-8'))
        except Exception as e:
            raise ValueError(f"Không thể đọc file {train_filepath}. Lỗi: {e}")

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
        # Gán nhãn padding là -100 (để CrossEntropyLoss bỏ qua)
        self.tag2idx[self.pad_tag] = -100 
        self.idx2tag = {i: t for t, i in self.tag2idx.items()}
    
    def encode_tokens(self, tokens: list[str]) -> torch.Tensor:
        ids = []
        for token in tokens:
            token = token.lower()
            ids.append(self.word2idx.get(token, self.word2idx[self.unk]))
        return torch.tensor(ids, dtype=torch.long)
    
    def encode_tags(self, tags: list[str]) -> torch.Tensor:
        ids = [self.tag2idx.get(tag, -100) for tag in tags] # Thêm get để an toàn
        return torch.tensor(ids, dtype=torch.long)
    
    # 3. THÊM: Hàm decode để dùng khi in kết quả kiểm tra
    def decode_tags(self, ids: list[int]) -> list[str]:
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return [self.idx2tag.get(i, "PAD") for i in ids]

    @property
    def n_tags(self) -> int:
        # Trả về số lượng tags thực tế (không tính padding -100)
        return len([i for i in self.tag2idx.values() if i != -100])
    
    @property
    def vocab_size(self) -> int:
        return len(self.word2idx)

class PhoNER_COVID19(Dataset):
    def __init__(self, path: str, vocab: Vocab):
        super().__init__()
        self.vocab = vocab
        # Thêm encoding='utf-8' để tránh lỗi trên Windows/Colab
        self.data = json.load(open(path, encoding='utf-8'))
    
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

def collate_fn(items: list[dict]) -> dict:
    input_ids = [item["input_ids"] for item in items]
    tags_ids = [item["tags_ids"] for item in items]
    
    # Tính lengths cho pack_padded_sequence
    lengths = torch.tensor([len(seq) for seq in input_ids], dtype=torch.long)
    
    # Padding
    padded_input = pad_sequence(input_ids, batch_first=True, padding_value=0)
    padded_tags = pad_sequence(tags_ids, batch_first=True, padding_value=-100)

    return {
        "input_ids": padded_input,
        "tags_ids": padded_tags,
        "lengths": lengths
    }
