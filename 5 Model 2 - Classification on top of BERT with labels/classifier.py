from experimenter import *

CONFIG = {
    'score': 'pmi_score', # 'mi_score', 'pmi_score'
    'labels_pred_hidden_layers_config': [768, 768], # a list of hidden layers dims
    'interest_pred_hidden_layers_config': [768, 768], # a list of hidden layers dims
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

    # encode the labels
    labels = get_all_possible_labels(input_df)
    le = LabelEncoder() # encode labels from string ('GPE', 'ORG', ...) to ordinal numbers (0, 1, ...)
    le.fit(labels)
    input_df['label1'] = le.transform(input_df['label1'])
    input_df['label2'] = le.transform(input_df['label2'])

    output_dir = f'{result_dir}\\{exp_index}'

    experiment_settings_str = ','.join([f'{key}={value}' for key, value in CONFIG.items()])
    print('running: ' + experiment_settings_str)
    experiment_results = run_experiment(input_df, CONFIG['score'],
                                        CONFIG['labels_pred_hidden_layers_config'], CONFIG['interest_pred_hidden_layers_config'],
                                        CONFIG['learning_rate'], CONFIG['batch_size'],
                                        CONFIG['num_epochs'], le, output_dir)
    log_experiment(experiment_log_file_path, CONFIG_HEADER, CONFIG, RESULTS_HEADER, experiment_results)