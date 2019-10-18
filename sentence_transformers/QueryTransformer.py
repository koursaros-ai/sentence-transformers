import torch
from torch import nn
from collections import OrderedDict
from typing import List
import os
import numpy as np
from tqdm import tqdm

from . import SentenceTransformer

class QueryTransformer(nn.Module):

    def __init__(self, sentence_transformer : SentenceTransformer, embedding_size, path=None):
        super().__init__()
        self.sentence_transformer = sentence_transformer
        self.linear = torch.nn.Linear(embedding_size, embedding_size)
        if path:
            self.load(path)

    def forward(self, features):
        output = self.sentence_transformer(features)['sentence_embedding']
        features.update({'sentence_embedding': self.linear(output)})
        return features

    def save(self, path):
        save_path = os.path.join(path, 'Query')
        os.makedirs(save_path, exist_ok=True)
        torch.save(self.linear.state_dict(), os.path.join(save_path, 'pytorch_model.bin'))

    def load(self, path):
        load_path = os.path.join(path, 'Query', 'pytorch_model.bin')
        self.linear.load_state_dict(torch.load(load_path))

    def encode(self, sentences: List[str], batch_size: int = 8, show_progress_bar: bool = None):
        """
       Computes sentence embeddings

       :param sentences:
           the sentences to embed
       :param batch_size:
           the batch size used for the computation
       :param show_progress_bar:
            Output a progress bar when encode sentences
       :return:
           a list with ndarrays of the embeddings for each sentence
       """

        all_embeddings = []
        length_sorted_idx = np.argsort([len(sen) for sen in sentences])

        iterator = range(0, len(sentences), batch_size)
        if show_progress_bar:
            iterator = tqdm(iterator, desc="Batches")

        for batch_idx in iterator:
            batch_tokens = []

            batch_start = batch_idx
            batch_end = min(batch_start + batch_size, len(sentences))

            longest_seq = 0

            for idx in length_sorted_idx[batch_start: batch_end]:
                sentence = sentences[idx]
                tokens = self.sentence_transformer.tokenize(sentence)
                longest_seq = max(longest_seq, len(tokens))
                batch_tokens.append(tokens)

            features = {}
            for text in batch_tokens:
                sentence_features = self.sentence_transformer.get_sentence_features(text, longest_seq)

                for feature_name in sentence_features:
                    if feature_name not in features:
                        features[feature_name] = []
                    features[feature_name].append(sentence_features[feature_name])

            for feature_name in features:
                features[feature_name] = torch.tensor(np.asarray(features[feature_name])).to(self.sentence_transformer.device)

            with torch.no_grad():
                embeddings = self.forward(features)
                embeddings = embeddings['sentence_embedding'].to('cpu').numpy()
                all_embeddings.extend(embeddings)

        reverting_order = np.argsort(length_sorted_idx)
        all_embeddings = [all_embeddings[idx] for idx in reverting_order]

        return all_embeddings