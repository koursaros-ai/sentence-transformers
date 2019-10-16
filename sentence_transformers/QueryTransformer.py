import torch
from torch import nn
from collections import OrderedDict
import os

from . import SentenceTransformer

class QueryTransformer(nn.Module):

    def __init__(self, sentence_transformer : SentenceTransformer, embedding_size):
        super().__init__()
        self.sentence_transformer = sentence_transformer
        self.linear = torch.nn.Linear(embedding_size, embedding_size)

    def forward(self, features):
        output = self.sentence_transformer(features)['sentence_embedding']
        features.update({'sentence_embedding': self.linear(output)})
        return features

    def save(self, path):
        save_path = os.path.join(path, 'QueryTransformer')
        self.save(save_path)