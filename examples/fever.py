from sentence_transformers import SentenceTransformer, losses
from sentence_transformers.readers import STSDataReader, FEVERReader
from sentence_transformers.datasets import SentencesDataset
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from torch.utils.data import DataLoader

def main():
    model = SentenceTransformer('bert-large-nli-mean-tokens')
    train_batch_size = 1
    num_epochs = 1
    warmup_steps = 100
    model_save_path = './fever-output'
    reader = FEVERReader()
    train_examples = reader.get_examples(table='test.scorer_train_title')
    train_data = SentencesDataset(train_examples, model)
    train_dataloader = DataLoader(train_data, shuffle=True, batch_size=train_batch_size)
    train_loss = losses.CosineSimilarityLoss(model=model)

    dev_examples = reader.get_examples(table='test.scorer_test_title')
    dev_data = SentencesDataset(examples=dev_examples, model=model)
    dev_dataloader = DataLoader(dev_data, shuffle=False, batch_size=train_batch_size)
    evaluator = EmbeddingSimilarityEvaluator(dev_dataloader)

    model.fit(train_objectives=[(train_dataloader, train_loss)],
              evaluator=evaluator,
              epochs=num_epochs,
              evaluation_steps=1000,
              warmup_steps=warmup_steps,
              output_path=model_save_path)


if __name__ == '__main__':
    main()