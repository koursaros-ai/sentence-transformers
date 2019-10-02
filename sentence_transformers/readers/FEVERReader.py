from . import InputExample
import csv
import gzip
import os
import psycopg2


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

class FEVERReader:

    def __init__(self, min_score=0.0,
                 max_score=1.0,
                 normalize_scores=True,
                 s1_idx=0,
                 s2_idx=1,
                 score_idx=2):
        self.min_score = min_score
        self.max_score = max_score
        self.normalize_scores = normalize_scores
        self.s1_idx = s1_idx
        self.s2_idx = s2_idx
        self.score_idx = score_idx

    def get_examples(self, split, filename=None, table=None, max_examples=0,):
        """
        filename specified which data split to use (train.csv, dev.csv, test.csv).
        """
        print('trying to connect to postgres...')
        conn = psycopg2.connect(POSTGRES_DSN)
        cur = conn.cursor()
        print('connected to postgres')
        limit = 'limit 500' if split == 'dev' else 'limit 5000'
        cur.execute(f'''
                select * from {table} order by random() {limit}
                ''')
        res = cur.fetchall()
        print('downloading data')
        examples = []
        for id, row in enumerate(res):
            score = float(row[self.score_idx])
            if self.normalize_scores:  # Normalize to a 0...1 value
                score = (score - self.min_score) / (self.max_score - self.min_score)
            s1 = row[self.s1_idx]
            s2 = row[self.s2_idx]
            examples.append(InputExample(guid=str(id), texts=[s1, s2], label=score))

        return examples



