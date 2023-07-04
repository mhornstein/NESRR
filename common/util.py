import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn

def results_to_files(results_dict, output_dir):
    results_df = pd.DataFrame(results_dict).set_index('epoch')

    for measurement in results_df.columns:
        results_df[measurement].plot(title=measurement.replace('_', ' '))
        plt.savefig(f'{output_dir}/{measurement}.jpg')
        plt.cla()

    results_df.to_csv(f'{output_dir}/results.csv', index=True)

def create_network(input_dim, hidden_layers_config, output_dim):
    '''
    Creates a deep model according to the given configuration
    :param input_dim: the input dimension of the network
    :param hidden_layers_config: indicates the hidden layers configuration of the network. \
                                 its format: [hidden_dim_1, dropout_rate_1, hidden_dim_2, dropout_rate_2, ...]. \
                                 for no dropout layer, use None value.
    :param output_dim: the output dimension of the network
    :return: the created model
    '''
    layers = []

    prev_dim = input_dim
    num_layers = len(hidden_layers_config) // 2  # Each layer has a dim and dropout rate

    for i in range(num_layers):
        dim = hidden_layers_config[i * 2]
        dropout_rate = hidden_layers_config[i * 2 + 1]

        layers.append(nn.Linear(prev_dim, dim))

        if dropout_rate is not None:
            layers.append(nn.Dropout(dropout_rate))

        layers.append(nn.ReLU())

        prev_dim = dim

    layers.append(nn.Linear(prev_dim, output_dim))

    model = nn.Sequential(*layers)

    return model