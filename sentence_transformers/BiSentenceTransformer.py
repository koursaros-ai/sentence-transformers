from torch import nn
from sentence_transformers import SentenceTransformer
import logging
import os
from typing import List, Dict, Tuple, Iterable, Type
import pytorch_transformers
import torch
from torch import nn, Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
import numpy as np

from .evaluation import SentenceEvaluator
from .util import import_from_string, batch_to_device, http_get


class BiSentenceTransformer(nn.Module):

    def __init__(self, model_a : SentenceTransformer, model_b : SentenceTransformer):
        super().__init__()
        self.model_a = model_a
        self.model_b = model_b


    def forward(self, features):
        sent_a, sent_b = features
        features_a = self.model_a(sent_a)['sentence_embedding']
        features_b = self.model_b(sent_b)
        features.update({'embedding_a' : features_a, 'embedding_b' : features_b})
        return features

    def fit(self,
            train_objective: Tuple[DataLoader, nn.Module],
            evaluator: SentenceEvaluator,
            epochs: int = 1,
            scheduler: str = 'WarmupLinear',
            warmup_steps: int = 10000,
            optimizer_class: Type[Optimizer] = pytorch_transformers.AdamW,
            optimizer_params : Dict[str, object]= {'lr': 2e-5, 'eps': 1e-6, 'correct_bias': False},
            weight_decay: float = 0.01,
            evaluation_steps: int = 0,
            output_path_base: str = None,
            save_best_model: bool = True,
            max_grad_norm: float = 1,
            fp16: bool = False,
            fp16_opt_level: str = '01',
            local_rank: int = -1
            ):
        """
        Train the model with the given training objective

        Each training objective is sampled in turn for one batch.
        We sample only as many batches from each objective as there are in the smallest one
        to make sure of equal training with each dataset.

        :param weight_decay:
        :param scheduler:
        :param warmup_steps:
        :param optimizer:
        :param evaluation_steps:
        :param output_path:
        :param save_best_model:
        :param max_grad_norm:
        :param fp16:
        :param fp16_opt_level:
        :param local_rank:
        :param train_objective:
            Tuple of DataLoader and LossConfig
        :param evaluator:
        :param epochs:
        """
        output_paths = [ output_path_base + str(i) for i in [0,1]]
        for path in output_paths:
            if path is not None:
                os.makedirs(path, exist_ok=True)
                if os.listdir(path):
                    raise ValueError("Output directory ({}) already exists and is not empty.".format(
                        path))

        dataloader, loss_model = train_objective

        dataloader.collate_fn = self.smart_batching_collate

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        loss_model.to(self.device)

        self.best_score = -9999

        min_batch_size = len(dataloader)
        num_train_steps = int(min_batch_size * epochs)

        # Prepare optimizers
        optimizers = []
        schedulers = []
        param_optimizer = list(loss_model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        t_total = num_train_steps
        if local_rank != -1:
            t_total = t_total // torch.distributed.get_world_size()

        optimizer = optimizer_class(optimizer_grouped_parameters, **optimizer_params)
        scheduler = self._get_scheduler(optimizer, scheduler=scheduler, warmup_steps=warmup_steps, t_total=t_total)

        if fp16:
            try:
                from apex import amp
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

            model, optimizer = amp.initialize(loss_model, optimizer, opt_level=fp16_opt_level)
            loss_model = model
            optimizer = optimizer

        global_step = 0
        data_iterator = iter(dataloader)

        for epoch in trange(epochs, desc="Epoch"):

            training_steps = 0
            loss_model.zero_grad()
            loss_model.train()

            for step in trange(min_batch_size, desc="Iteration"):
                try:
                    data = next(data_iterator)
                except StopIteration:
                    logging.info("Restart data_iterator")
                    data_iterator = iter(dataloader)
                    data = next(data_iterator)

                print(data)
                features, labels = batch_to_device(data, self.device)
                loss_value = loss_model(features, labels)

                if fp16:
                    with amp.scale_loss(loss_value, optimizer) as scaled_loss:
                        scaled_loss.backward()
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), max_grad_norm)
                else:
                    loss_value.backward()
                    torch.nn.utils.clip_grad_norm_(loss_model.parameters(), max_grad_norm)

                training_steps += 1

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if evaluation_steps > 0 and training_steps % evaluation_steps == 0:
                    self._eval_during_training(evaluator, output_paths, save_best_model, epoch, training_steps)
                    loss_model.zero_grad()
                    loss_model.train()

            self._eval_during_training(evaluator, output_paths, save_best_model, epoch, -1)

    def _eval_during_training(self, evaluator, output_paths, save_best_model, epoch, steps):
        """Runs evaluation during the training"""
        if evaluator is not None:
            for output_path in output_paths:
                score = evaluator(self, output_path=output_path, epoch=epoch, steps=steps)
                if score > self.best_score and save_best_model:
                    self.save(output_path)
                    self.best_score = score

    def _get_scheduler(self, optimizer, scheduler: str, warmup_steps: int, t_total: int):
        """
        Returns the correct learning rate scheduler
        """
        scheduler = scheduler.lower()
        if scheduler == 'constantlr':
            return pytorch_transformers.ConstantLRSchedule(optimizer)
        elif scheduler == 'warmupconstant':
            return pytorch_transformers.WarmupConstantSchedule(optimizer, warmup_steps=warmup_steps)
        elif scheduler == 'warmuplinear':
            return pytorch_transformers.WarmupLinearSchedule(optimizer, warmup_steps=warmup_steps, t_total=t_total)
        elif scheduler == 'warmupcosine':
            return pytorch_transformers.WarmupCosineSchedule(optimizer, warmup_steps=warmup_steps, t_total=t_total)
        elif scheduler == 'warmupcosinewithhardrestarts':
            return pytorch_transformers.WarmupCosineWithHardRestartsSchedule(optimizer, warmup_steps=warmup_steps, t_total=t_total)
        else:
            raise ValueError("Unknown scheduler {}".format(scheduler))

    def smart_batching_collate(self, batch):
        """
        Transforms a batch from a SmartBatchingDataset to a batch of tensors for the model

        :param batch:
            a batch from a SmartBatchingDataset
        :return:
            a batch of tensors for the model
        """
        num_texts = len(batch[0][0])

        labels = []
        paired_texts = [[] for _ in range(num_texts)]
        max_seq_len = [0] * num_texts
        for tokens, label in batch:
            labels.append(label)
            for i in range(num_texts):
                paired_texts[i].append(tokens[i])
                max_seq_len[i] = max(max_seq_len[i], len(tokens[i]))

        features = []
        for idx in range(num_texts):
            max_len = max_seq_len[idx]
            feature_lists = {}
            for text in paired_texts[idx]:
                sentence_features = self.get_sentence_features(text, max_len)

                for feature_name in sentence_features:
                    if feature_name not in feature_lists:
                        feature_lists[feature_name] = []
                    feature_lists[feature_name].append(sentence_features[feature_name])

            for feature_name in feature_lists:
                feature_lists[feature_name] = torch.tensor(np.asarray(feature_lists[feature_name]))

            features.append(feature_lists)

        return {'features': features, 'labels': torch.stack(labels)}

    def get_sentence_features(self, *features):
        return self.model_a.get_sentence_features(*features)