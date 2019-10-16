import torch
from torch import nn
from collections import OrderedDict

from . import SentenceTransformer

class QueryTransformer(nn.Module):

    def __init__(self, sentence_transformer : SentenceTransformer, embedding_size):
        self.sentence_transformer = sentence_transformer
        self.linear = torch.nn.Linear(embedding_size, embedding_size)
        super().__init__()

    def forward(self, features):
        output = self.sentence_transformer(features)['sentence_embedding']
        features.update({'sentence_embedding': self.linear(output)})
        return features