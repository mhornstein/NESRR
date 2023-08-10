import sys

sys.path.append('..\\')
from common.util import *

CONFIG_HEADER = ['exp_index', 'score', 'threshold']
RESULTS_HEADER = ['train_acc', 'val_acc', 'test_acc']

if __name__ == '__main__':
    if len(sys.argv) < 3:
        raise ValueError(f'Not enough arguments. Arguments given: {sys.argv[1:]}')

    input_file = sys.argv[1] # e.g. '../data/dummy/dummy_data.csv' or '../data/data.csv'
    result_dir = sys.argv[2] # e.g. 'results'

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    experiment_log_file_path = f'{result_dir}\\experiments_logs.csv'
    exp_index = init_experiment_log_file(experiment_log_file_path, CONFIG_HEADER, RESULTS_HEADER)