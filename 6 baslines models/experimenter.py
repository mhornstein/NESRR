import sys
from sklearn.model_selection import train_test_split
from common.util import *
from common.classifier_util import *
from sklearn.metrics import accuracy_score
import spacy

sys.path.append('..\\')

nlp = spacy.load("en_core_web_lg", disable=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer"])
MASK_LABEL = '[MASK]'
HEURISTICS = ['entities_count', 'mask_dist', 'random_score']

CONFIG_HEADER = ['exp_index', 'score', 'heuristic', 'heuristic_threshold']
RESULTS_HEADER = ['train_acc', 'val_acc', 'test_acc']

def get_heuristics_scores(s):
    sent = s['masked_sent']
    doc = nlp(sent)

    entities_count = len(doc.ents)

    mask_token_indices = [token.i for token in doc if token.text == 'MASK']
    mask_dist = mask_token_indices[1] - mask_token_indices[0] - 3 # substracting [ and ] that are also considered tokens

    random_score = random.choice([0, 1])

    return entities_count, mask_dist, random_score

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

    df[HEURISTICS] = df.apply(get_heuristics_scores, axis=1, result_type='expand')

    for score in ['mi_score', 'pmi_score']:
        for heuristic in HEURISTICS:
            X_train, X_tmp, y_train, y_tmp = train_test_split(df[heuristic], df[score], random_state=42, test_size=0.4)
            y_train, y_tmp = score_to_label(y_train, y_tmp)

            X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, random_state=42, test_size=0.5)

            min_heuristic_score = X_train.min()
            max_heuristic_score = X_train.max()

            for heuristic_threshold in range(min_heuristic_score, max_heuristic_score):
                experiment_settings = {
                    'exp_index': exp_index,
                    'score': score,
                    'heuristic': heuristic,
                    'heuristic_threshold': heuristic_threshold
                }

                experiment_settings_str = ','.join([f'{key}={value}' for key, value in experiment_settings.items()])
                print('running: ' + experiment_settings_str)

                train_predictions = (X_train > heuristic_threshold).astype(int).values
                val_predictions = (X_val > heuristic_threshold).astype(int).values
                test_predictions = (X_test > heuristic_threshold).astype(int).values

                experiment_results = {'train_acc': accuracy_score(y_train, train_predictions),
                                      'val_acc': accuracy_score(y_val, val_predictions),
                                      'test_acc': accuracy_score(y_test, test_predictions)}

                log_experiment(experiment_log_file_path, CONFIG_HEADER, experiment_settings, RESULTS_HEADER, experiment_results)
                exp_index += 1