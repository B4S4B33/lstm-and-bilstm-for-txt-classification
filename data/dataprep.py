import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import re

class AGNewsDataset(Dataset):
    def __init__(self, csv_path, vocab=None, max_len=200, min_freq=2):
        df = pd.read_csv(csv_path, header=None, names=['label', 'title', 'description'])
        
        # Filter out rows with non-numeric labels (like headers) and convert to integer
        df = df[pd.to_numeric(df['label'], errors='coerce').notnull()].copy()
        df['label'] = pd.to_numeric(df['label']).astype(int)
        
        # Combine title + description
        self.texts = (df['title'].astype(str) + ' ' + df['description'].astype(str)).tolist()
        self.labels = (df['label'] - 1).tolist()  # 0-indexed: 0,1,2,3

        # Build vocab
        if vocab is None:
            self.build_vocab(min_freq)
        else:
            self.vocab = vocab
            self.vocab_size = len(vocab)

        self.max_len = max_len
        self.encoded_texts = [self.encode(text) for text in self.texts]

    def build_vocab(self, min_freq):
        counter = Counter()
        for text in self.texts:
            tokens = self.tokenize(text)
            counter.update(tokens)

        # Special tokens
        self.vocab = {'<pad>': 0, '<unk>': 1}
        for token, freq in counter.items():
            if freq >= min_freq:
                self.vocab[token] = len(self.vocab)
        self.vocab_size = len(self.vocab)

    def tokenize(self, text):
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        return text.split()

    def encode(self, text):
        tokens = self.tokenize(text)
        ids = [self.vocab.get(t, 1) for t in tokens]  # 1 = <unk>
        if len(ids) > self.max_len:
            ids = ids[:self.max_len]
        else:
            ids += [0] * (self.max_len - len(ids))  # 0 = <pad>
        return ids

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.tensor(self.encoded_texts[idx], dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)

# Usage
train_dataset = AGNewsDataset('/content/train.csv')
test_dataset = AGNewsDataset('/content/test.csv', vocab=train_dataset.vocab)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
print(f"Vocab size: {train_dataset.vocab_size}")
print(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")