from torch.nn.functional import cosine_similarity
from torch import nn, Tensor
from typing import Union, Tuple, List, Iterable, Dict
from ..BiSentenceTransformer import BiSentenceTransformer


class BiCosineSimilarityLoss(nn.Module):

    def __init__(self, model: BiSentenceTransformer):
        super(BiCosineSimilarityLoss, self).__init__()
        self.model = model

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        rep_a, rep_b = self.model(sentence_features)

        output = cosine_similarity(rep_a, rep_b)
        loss_fct = nn.MSELoss()
        if labels is not None:
            loss = loss_fct(output, labels.view(-1))
            return loss
        else:
            return [rep_a, rep_b], output