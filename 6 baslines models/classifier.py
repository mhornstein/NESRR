import sys

sys.path.append('..\\')
from common.util import *
import spacy

nlp = spacy.load("en_core_web_lg", disable=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer"])
MASK_LABEL = '[MASK]'

CONFIG_HEADER = ['exp_index', 'score', 'threshold']
RESULTS_HEADER = ['train_acc', 'val_acc', 'test_acc']

def get_scores(s):
    sent = s['masked_sent']
    doc = nlp(sent)

    ent_count = len(doc.ents)

    mask_token_indices = [token.i for token in doc if token.text == 'MASK']
    mask_dist = mask_token_indices[1] - mask_token_indices[0] - 3 # substracting [ and ] that are also considered tokens

    random_score = random.choice([0, 1])

    return ent_count, mask_dist, random_score

if __name__ == '__main__':
    if len(sys.argv) < 3:
        raise ValueError(f'Not enough arguments. Arguments given: {sys.argv[1:]}')

    input_file = sys.argv[1] # e.g. '../data/dummy/dummy_data.csv' or '../data/data.csv'
    result_dir = sys.argv[2] # e.g. 'results'

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    experiment_log_file_path = f'{result_dir}\\experiments_logs.csv'
    exp_index = init_experiment_log_file(experiment_log_file_path, CONFIG_HEADER, RESULTS_HEADER)

    df = pd.read_csv(input_file)
    df = df.set_index('sent_id')

    df[['ent_count', 'mask_dist', 'random_score']] = df.apply(get_scores, axis=1, result_type='expand')

    print()
