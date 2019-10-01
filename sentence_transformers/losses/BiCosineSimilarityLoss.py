import torch
from torch import nn, Tensor
from typing import Union, Tuple, List, Iterable, Dict
from ..SentenceTransformer import SentenceTransformer

class BiCosineSimilarityLoss(nn.Module):
    def __init__(self, model: SentenceTransformer):
        super(BiCosineSimilarityLoss, self).__init__()
        self.model = model


    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
       features = self.model(sentence_features)
       rep_a = features['embedding_a']
       rep_b = features['embedding_b']

       output = torch.cosine_similarity(rep_a, rep_b)
       loss_fct = nn.MSELoss()
       if labels is not None:
           loss = loss_fct(output, labels.view(-1))
           return loss
       else:
           return [rep_a, rep_b], output