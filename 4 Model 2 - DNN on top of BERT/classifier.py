from experimenter import *

CONFIG = {
    'score': 'pmi_score', # 'mi_score', 'pmi_score'
    'score_threshold_type': 'percentile', # 'percentile', 'std_dist'
    'score_threshold_value': 0.5, # 0.25, 0.5, 0.75 for precentile; -1, -2 for pmi std; 1, 2 for mi std
    'hidden_layers_config': [786,0.5,786,0.5,786], # a list with hidden-dim->dropout-rate->hidden-dim->dropout-rate->...
    'learning_rate': 0.0001,
    'batch_size': 128,
    'num_epochs': 5
}

if __name__ == '__main__':
    if len(sys.argv) < 4:
        raise ValueError(f'Not enough arguments. Arguments given: {sys.argv[1:]}')

    input_file = sys.argv[1]  # e.g. '../data/dummy/dummy_data.csv' or '../data/data.csv'
    embeddings_file = sys.argv[2] # e.g. '../data/dummy/embeddings_dummy.out' or '../data/embeddings.out'
    result_dir = sys.argv[3] # e.g. 'results'

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    experiment_log_file_path = f'{result_dir}\\experiments_logs.csv'
    exp_index = init_experiment_log_file(experiment_log_file_path, CONFIG_HEADER, RESULTS_HEADER)

    CONFIG['exp_index'] = exp_index

    input_df = create_df(input_file, embeddings_file)

    output_dir = f'{result_dir}\\{exp_index}'

    experiment_settings_str = ','.join([f'{key}={value}' for key, value in CONFIG.items()])
    print('running: ' + experiment_settings_str)
    experiment_results = run_experiment(input_df, CONFIG['score'], CONFIG['score_threshold_type'], CONFIG['score_threshold_value'],
                                        CONFIG['hidden_layers_config'], CONFIG['learning_rate'], CONFIG['batch_size'],
                                        CONFIG['num_epochs'], output_dir)
    log_experiment(experiment_log_file_path, CONFIG_HEADER, CONFIG, RESULTS_HEADER, experiment_results)