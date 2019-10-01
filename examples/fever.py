from sentence_transformers import SentenceTransformer, losses, BiSentenceTransformer
from sentence_transformers.readers import STSDataReader, FEVERReader
from sentence_transformers.datasets import SentencesDataset
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from torch.utils.data import DataLoader
import os
from numpy import dot
from numpy.linalg import norm
import time
import psycopg2
from psycopg2.extras import execute_values

HOST = '54.196.150.193'
USER = 'postgres'
PASS = os.environ.get('PG_PASS')
PGSSLROOTCERT = os.environ.get('PGSSLROOTCERT')
if PASS == None or PGSSLROOTCERT == None:
    print("Please set PG_PASS and PGSSLROOTCERT env variable")
    raise SystemExit()
DBNAME = 'fever'
os.environ['PGSSLMODE'] = 'verify-ca'
__location__ = os.path.dirname(__file__)
POSTGRES_DSN = f'''dbname='fever' user='{USER}' host='{HOST}' password='{PASS}' '''
BATCH_SIZE = 1000

def main():
    model_a = SentenceTransformer('bert-large-nli-mean-tokens')
    model_b = SentenceTransformer('bert-large-nli-mean-tokens')
    model = BiSentenceTransformer(model_a, model_b)
    train_batch_size = 2
    num_epochs = 1
    warmup_steps = 100
    model_save_path = './fever-output'
    reader = FEVERReader()
    train_examples = reader.get_examples('train',table='test.scorer_train_title')
    train_data = SentencesDataset(train_examples, model_a)
    train_dataloader = DataLoader(train_data, shuffle=True, batch_size=train_batch_size)
    train_loss = losses.BiCosineSimilarityLoss(model)

    dev_examples = reader.get_examples('dev',table='test.scorer_test_title')
    dev_data = SentencesDataset(examples=dev_examples, model=model)
    dev_dataloader = DataLoader(dev_data, shuffle=False, batch_size=train_batch_size)
    evaluator = EmbeddingSimilarityEvaluator(dev_dataloader)

    model.fit(train_objective=(train_dataloader, train_loss),
              evaluator=evaluator,
              epochs=num_epochs,
              evaluation_steps=5000,
              warmup_steps=warmup_steps,
              output_path_base=model_save_path)

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
        left join test.knn_benchmark b on b.claim_id = c.id 
        and b.line_number = l.line_number and b.article_id = l.article_id
        where is_test_set and b.claim_id is null and verifiable
        ''')
    res = cur.fetchall()

    # batch = (claim, evidence, id, article_id, line_number)
    def score(batch):
        claims = [x[0] for x in batch]
        evidences = [x[1] for x in batch]
        start = time.time()
        claim_vectors = model.encode(claims)
        print(f'encoded claims in {time.time() - start} seconds')
        state = time.time()
        ev_vectors = model.encode(evidences)
        print(f'encoded evidence in {time.time() - start} seconds')
        start = time.time()
        scores = [float(dot(claim, evidence)) / (norm(claim) * norm(evidence))
                  for claim, evidence in zip(claim_vectors, ev_vectors) ]
        print(f'scored in {time.time() - start} seconds')
        return [(*b[2:], s) for b, s in zip(batch, scores)]

    to_dump = score(res)
    buffer = []

    for i, row in enumerate(to_dump):
        buffer.append(row)
        if i and i % BATCH_SIZE == 0:
            execute_values(cur, 'INSERT INTO test.knn_benchmark VALUES %s', buffer)
            conn.commit()
            buffer.clear()
    if len(buffer) > 0:
        execute_values(cur, 'INSERT INTO test.knn_benchmark VALUES %s', buffer)
        conn.commit()
        buffer.clear()
    conn.close()


if __name__ == '__main__':
    # main()
    eval()

    # CONVERT TO TENSORFLOW FOR BERT_AS_A_SERVICE?
    # from pytorch_transformers.convert_pytorch_checkpoint_to_tf import convert_pytorch_checkpoint_to_tf
    #
    # model = BertModel.from_pretrained("./data/finetuned_lm")
    # convert_pytorch_checkpoint_to_tf(model, "./data/finetuned_lm_tf", "fine_tuned_tf")