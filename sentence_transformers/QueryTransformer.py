import torch
from torch import nn
from collections import OrderedDict

from . import SentenceTransformer

class QueryTransformer(nn.Sequential):

    def __init__(self, sentence_transformer : SentenceTransformer, embedding_size):

        modules = [ ('sentence_transformer', sentence_transformer),
                    ('feedforward_layer' , torch.nn.Linear(embedding_size, embedding_size)) ]
        super().__init__(modules)