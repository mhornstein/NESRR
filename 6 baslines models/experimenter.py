import sys
from sklearn.model_selection import train_test_split
from common.util import *
from common.classifier_util import *
from sklearn.metrics import accuracy_score, classification_report
import spacy
import time

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

    mask_token_indices = [token.i for token in doc if 'MASK' in token.text]
    if len(mask_token_indices) == 1: # The two MASK placeholders appear in the same token. This occurs in cases like: [MASK]-to-[MASK] or [MASK]-[MASK]
        mask_dist = 0
    else:
        mask_dist = mask_token_indices[1] - mask_token_indices[0]

    random_score = random.choice([0, 1])

    return entities_count, mask_dist, random_score

if __name__ == '__main__':
    start_time = time.time()

    if len(sys.argv) < 4:
        raise ValueError(f'Not enough arguments. Arguments given: {sys.argv[1:]}')

    input_file = sys.argv[1] # e.g. '../data/dummy/dummy_data.csv' or '../data/data.csv'
    log_test_classification = eval(sys.argv[2]) # should be either True or False
    result_dir = sys.argv[3] # e.g. 'results'

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    experiment_log_file_path = f'{result_dir}\\experiments_logs.csv'
    exp_index = init_experiment_log_file(experiment_log_file_path, CONFIG_HEADER, RESULTS_HEADER)

    df = pd.read_csv(input_file)
    df = df.set_index('sent_id')

    print('Computing heuristics for each sentence...')

    df[HEURISTICS] = df.apply(get_heuristics_scores, axis=1, result_type='expand')

    print('Classifying the sentences according to each possible score, heuristic and threshold.')

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

                output_dir = f'{result_dir}\\{exp_index}'
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                experiment_settings_str = ','.join([f'{key}={value}' for key, value in experiment_settings.items()])
                print('running: ' + experiment_settings_str)

                train_predictions = (X_train > heuristic_threshold).astype(int).values
                val_predictions = (X_val > heuristic_threshold).astype(int).values
                test_predictions = (X_test > heuristic_threshold).astype(int).values

                experiment_results = {'train_acc': accuracy_score(y_train, train_predictions),
                                      'val_acc': accuracy_score(y_val, val_predictions),
                                      'test_acc': accuracy_score(y_test, test_predictions)}

                if log_test_classification == True:
                    sent_ids = X_test.index
                    out_df = pd.DataFrame({'sent_ids': sent_ids,
                                          'target_label': y_test,
                                          'predicted_label': test_predictions,
                                          'predicted_score': X_test})
                    out_df = out_df.set_index('sent_ids', drop=True)
                    out_df.index.name = None  # remove index column name

                    test_data = df.loc[sent_ids]
                    test_data = test_data[['label1', 'label2', 'ent1', 'ent2', 'masked_sent']]

                    out_df = pd.concat([out_df, test_data], axis=1)
                    out_df.to_csv(f'{output_dir}\\test_predictions_results.csv', index=True)

                test_classification_report = classification_report(y_test, test_predictions, zero_division=1)
                with open(f'{output_dir}\\test_report.txt', 'w') as file:
                    file.write(f'Test classification report:\n')
                    file.write(test_classification_report)

                log_experiment(experiment_log_file_path, CONFIG_HEADER, experiment_settings, RESULTS_HEADER, experiment_results)
                exp_index += 1

    print(f'Done. Total time: {time.time()-start_time} seconds.')