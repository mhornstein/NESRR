from experimenter import *

CONFIG = {
    'score': 'pmi_score', # 'mi_score', 'pmi_score'
    'score_threshold_type': 'percentile', # 'percentile', 'std_dist'
    'score_threshold_value': 0.5, # 0.25, 0.5, 0.75 for precentile; -1, -2 for pmi std; 1, 2 for mi std
    'hidden_layers_config': [786,0.5,786,0.5,786], # a list with hidden-dim->dropout-rate->hidden-dim->dropout-rate->...
    'learning_rate': 0.0001,
    'batch_size': 128,
    'num_epochs': 15
}

if __name__ == '__main__':
    result_dir = 'results'

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    log_file_path = f'{result_dir}\\experiments_logs.csv'
    exp_index = init_experiment_log_file(log_file_path, CONFIG_HEADER, RESULTS_HEADER)

    CONFIG['exp_index'] = exp_index

    embeddings_file = '..\\data\\embeddings.out' # '../data/dummy/embeddings_dummy.out'
    input_file = '..\\data\\data.csv' # '../data/dummy/dummy_data.csv'

    input_df = create_df(input_file, embeddings_file)

    output_dir = f'{result_dir}\\{exp_index}'

    settings_str = ','.join([f'{key}={value}' for key, value in CONFIG.items()])
    print('running: ' + settings_str)
    results = run_experiment(input_df, CONFIG['score'], CONFIG['score_threshold_type'], CONFIG['score_threshold_value'],
                             CONFIG['hidden_layers_config'], CONFIG['learning_rate'], CONFIG['batch_size'],
                             CONFIG['num_epochs'], output_dir)
    log_experiment(log_file_path, CONFIG_HEADER, CONFIG, RESULTS_HEADER, results)