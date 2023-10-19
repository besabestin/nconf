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
        #TODO: dict
        # word: idx
        # word2idx[word]: word

    #TODO: encode words, decode indices
    def encode(self, words):
        return [] 

    def decode(self, indices):
        return []

    def __len__(self):
        return len(self.vocab)


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
        #TODO: all input output possibilities
        #inputs i:0,le-cx-1
        #outputs i:1,len-cx
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
        return None, None