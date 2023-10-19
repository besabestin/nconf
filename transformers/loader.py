import torch
import torch.nn as nn
import torch.nn.functional as F
import os

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

torch.manual_seed(1345)

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.vocab = {}
    
    def create_vocab(self, full_text):
        self.vocab = set(full_text)
        self.word2idx = {word: idx for idx, word in enumerate(sorted(list(self.vocab)))}
        self.idx2word = {self.word2idx[word]: word for word in self.vocab}

    def encode(self, words):
        return [self.word2idx[word] for word in words]

    def decode(self, indices):
        return [self.idx2word[idx] for idx in indices]

    def __len__(self):
        return len(self.vocab)


# batch_size,
# context_size,
class Corpus(object):
    def __init__(self, path, batch_size=32, context_size=32) -> None:
        self.dictionary = Dictionary()
        self.words = []
        self.batch_size = batch_size
        self.context_size = context_size
        self.train_data = {}
        self.val_data = {}
        self.tokenize(path)
        self.prepare(self.words)


    def tokenize(self, path):
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r') as f:
            for line in f:
                words = line.split() + ['<eos>']
                self.words.extend(words)
        self.dictionary.create_vocab(self.words)


    def prepare(self, words):
        print(f'inspect words: {len(words)} {words[0:8]}')
        #OMIT: following two lines
        inputs = torch.stack([torch.tensor(self.dictionary.encode(words[i:i+self.context_size]), dtype=torch.long) for i in range(len(words) - self.context_size - 1)])
        outputs = torch.stack([torch.tensor(self.dictionary.encode(words[i:i+self.context_size]), dtype=torch.long) for i in range(1,len(words) - self.context_size)])
        #randomize the order of the datasets
        _perm = torch.randperm(inputs.shape[0])
        train_split = int(0.8*inputs.shape[0])

        self.train_data = {
            'X': inputs[_perm][:train_split, :],
            'y': outputs[_perm][:train_split, :]
        }
        self.val_data = {
            'X': inputs[_perm][train_split:, :],
            'y': outputs[_perm][train_split:, :]
        }
        
    #OMIT: whole get_batch function
    def get_batch(self, _stage):
        dataset = self.train_data if _stage == 'train' else self.val_data
        _perm = torch.randperm(dataset['X'].shape[0])
        return dataset['X'][_perm][:self.batch_size, :].to(device), dataset['y'][_perm][:self.batch_size, :].to(device)