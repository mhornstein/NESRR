import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn
import os
import csv
import random

NETWORK_SIZES = [64, 128, 256, 512]
def draw_hidden_layers_config():
    hidden_layers_config = random.sample(NETWORK_SIZES, random.randint(2, 4))
    return hidden_layers_config

def create_network(input_dim, hidden_layers_config, output_dim):
    '''
    Creates a deep model according to the given configuration
    :param input_dim: the input dimension of the network
    :param hidden_layers_config: indicates the hidden layers configuration of the network. Its format: [hidden_dim_1, hidden_dim_2, ...]
    :param output_dim: the output dimension of the network
    :return: the created model
    '''
    layers = []
    prev_dim = input_dim
    for dim in hidden_layers_config:
        layers.append(nn.Linear(prev_dim, dim))
        layers.append(nn.ReLU())
        prev_dim = dim
    layers.append(nn.Linear(prev_dim, output_dim))
    model = nn.Sequential(*layers)
    return model

def save_df_plot(df, title, output_dir):
    ax = df.plot(title=title)

    # Modify the legend labels so they have whitespaces instead of _
    handles, labels = ax.get_legend_handles_labels()
    labels = [label.replace('_', ' ') for label in labels]
    ax.legend(handles, labels)

    plt.savefig(f'{output_dir}/{title}.jpg')
    plt.cla()

    plt.close()

def results_to_files(results_df, output_dir):
    save_df_plot(df=results_df[['avg_train_loss', 'avg_val_loss']], title='loss', output_dir=output_dir)
    save_df_plot(df=results_df[['avg_train_acc', 'avg_val_acc']], title='accuracy', output_dir=output_dir)
    if 'avg_val_labels_acc' in results_df.columns:
        save_df_plot(df=results_df[['avg_train_labels_acc', 'avg_val_labels_acc']], title='labels-accuracy', output_dir=output_dir)
    save_df_plot(df=results_df[['epoch_time']], title='epochs-time', output_dir=output_dir)

    results_df.to_csv(f'{output_dir}/train_logs.csv', index=True)

def init_experiment_config_file(file_path, config_header):
    if not os.path.exists(file_path):
        with open(file_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(config_header)

def write_experiment_config(file_path, entry, config_header):
    esc_value = lambda val: str(val).replace(',', '') # remove commas from values' conent, so csv format won't be damaged
    with open(file_path, 'a') as file:
        values = [esc_value(entry[key]) for key in config_header]
        file.write(','.join(str(value) for value in values) + '\n')

def init_experiment_log_file(file_path, config_header, results_header):
    '''
    This function initializes the experiment log file if none exists.
    If the file already exists, it reads it and fetches the last experiment index out of it,
    to continue the experiments indexing from that point.
    Finally, the function returns the experiment index for further use.
    '''
    if not os.path.exists(file_path):
        exp_index = 1
        with open(file_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(config_header + results_header)
    else:
        df = pd.read_csv(file_path, index_col='exp_index')
        if len(df) == 0: # the log file was just initialized but never used
            exp_index = 1
        else:
            exp_index = df.index[-1] + 1
    return exp_index

def log_experiment(file_path, config_header, experiment_config, results_header, experiment_results):
    esc_value = lambda val: str(val).replace(',', '') # remove commas from values' conent, so csv format won't be damaged
    with open(file_path, 'a') as file:
        values = [esc_value(experiment_config[key]) for key in config_header]
        values += [experiment_results[key] for key in results_header]
        file.write(','.join(str(value) for value in values) + '\n')

def get_all_possible_labels(df):
    label1_values = set(df['label1'].unique())
    label2_values = set(df['label2'].unique())
    labels = list(label1_values | label2_values)
    return labels