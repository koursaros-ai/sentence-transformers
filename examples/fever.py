from sentence_transformers import SentenceTransformer, losses
from sentence_transformers.readers import STSDataReader, FEVERReader
from sentence_transformers.datasets import SentencesDataset
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from torch.utils.data import DataLoader
import os
from numpy import dot
from numpy.linalg import norm
import time
import psycopg2

HOST = '54.196.150.193'
USER = 'postgres'
PASS = os.environ.get('PG_PASS')
PGSSLROOTCERT = os.environ.get('PGSSLROOTCERT')
if PASS == None or PGSSLROOTCERT == None:
    print("Please set PG_PASS and CERT_PATH env variable")
    raise SystemExit()
DBNAME = 'fever'
os.environ['PGSSLMODE'] = 'verify-ca'
__location__ = os.path.dirname(__file__)
POSTGRES_DSN = f'''dbname='fever' user='{USER}' host='{HOST}' password='{PASS}' '''

def main():
    model = SentenceTransformer('bert-large-nli-mean-tokens')
    train_batch_size = 2
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

def eval():
    model = SentenceTransformer('./fever-output')

    print('trying to connect to postgres...')
    conn = psycopg2.connect(POSTGRES_DSN)
    cur = conn.cursor()
    print('connected to postgres')
    cur.execute(f'''
        select claim, concat(a.title, ', ', l.text), c.id, l.article_id, l.line_number
        from sets.claims c
        join test.elastic el on c.id = el.claim_id
        join wiki.articles a on a.fever_id = any(el.fever_ids)
        join wiki.lines l on l.article_id = a.id
        and l.line_number = b.line_number and l.article_id = b.article_id
        where b.claim_id is null and is_blind_set
        limit 1000
        ''')
    res = cur.fetchall()

    # batch = (claim, evidence, id, article_id, line_number)
    def score(batch):
        claims = [x[0] for x in batch]
        evidences = [x[1] for x in batch]
        claim_vectors = model.encode(claims)
        ev_vectors = model.encode(evidences)
        scores = [dot(claim, evidence) / (norm(claim) * norm(evidence))
                  for claim, evidence in zip(claim_vectors, ev_vectors) ]
        return [(*b[2:], s) for b, s in zip(batch, scores)]

    to_dump = score(res)
    for row in to_dump:
        cur.execute('INSERT INTO test.knn_benchmark VALUES (%s, %s, %s, %s)', row)
    cur.commit()
    cur.close()




if __name__ == '__main__':
    eval()
    # main()