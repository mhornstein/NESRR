"""
This script trains a deep network model for a regression task that focuses on predicting the mutual information score between two masked entities within a sentence.
It requires a dataset containing the masked sentences and the sentence embeddings generated by a BERT model as input.

Usage:
  regressor.py --input_file=<file> --embedding_file=<file> [--mi_trans=<mi_transformation>] [--reg_network=<network_config>] [--output_dir=<directory>] [--epochs=<count>] [--batch_size=<size>] [--lr=<rate>]
  regressor.py (-h | --help)

Options:
  -h --help                         Show this screen.
  --input_file=<file>               Path to the dataset file.
  --embedding_file=<file>           Path to the embedding file.
  --mi_trans=<mi_transformation>    The transformation to apply to the mutual information scores.
                                    Available options: None, minmax, ln, sqrt. [default: None]
  --reg_network=<network_config>    The hidden layers for the network. The format of the list should be as follows: [hidden_dim_1, dropout_rate_1, hidden_dim_2, dropout_rate_2, ...]. Use 'None' for the dropout layer to indicate no dropout layer for that particular hidden layer. [default: [512, None, 128, None]]
  --output_dir=<directory>          Path to the directory for writing the results. [default: ./results]
  --epochs=<count>                  Number of epochs for training. [default: 10]
  --batch_size=<size>               Batch size. [default: 32]
  --lr=<rate>                       Learning rate for training. [default: 1e-5]
"""


from sklearn.model_selection import train_test_split
from transformers.optimization import AdamW
import torch
from torch.utils.data import TensorDataset, DataLoader
import time
import warnings
warnings.filterwarnings("ignore", category=FutureWarning) # Disable the warning
import os
import sys
from docopt import docopt

sys.path.append('../')
from common.regressor_util import *

BERT_OUTPUT_SHAPE = 768

####################

class BERT_Regressor(nn.Module):

    def __init__(self, input_dim, hidden_layers_config):
        '''
        :param input_dim: the input dimension of the network
        :param hidden_layers_config: indicates the hidden layers configuration of the network. \
                                     its format: [hidden_dim_1, dropout_rate_1, hidden_dim_2, dropout_rate_2, ...]. \
                                     for no dropout layer, use None value.
        '''
        super(BERT_Regressor, self).__init__()
        self.model = create_network(input_dim=input_dim, hidden_layers_config=hidden_layers_config, output_dim=1)

    def forward(self, x):
        return self.model(x)

####################

def create_df(data_file, embs_file, mi_transformation):
    '''
    Creates a dataframe that consists of the sentences embeddings and their respective (transformed) mi_score
    '''
    # Load input data file
    df = pd.read_csv(data_file)
    df['mi_score'] = transform_mi(df['mi_score'], mi_transformation)

    df = df.set_index('sent_id')

    # Load embeddings file
    embs = pd.read_csv(embs_file, sep=' ', header=None)
    embs = embs.set_index(embs.columns[0])  # set sentence id as the index of the dataframe

    # combine both for a single df
    df = pd.concat([embs, df], axis=1)

    return df

def create_data_loader(X, y, batch_size, shuffle):
    ids_tensor = torch.tensor(X.index, dtype=torch.int64).unsqueeze(dim=1).to(device)
    X_tensor = torch.tensor(X.values, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y.values.reshape(-1, 1), dtype=torch.float32).to(device)
    dataset = TensorDataset(ids_tensor, X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get command line arguments
    arguments = docopt(__doc__)

    input_file = arguments['--input_file']
    embedding_file = arguments['--embedding_file']
    mi_transformation = arguments['--mi_trans']
    reg_network = eval(arguments['--reg_network'])
    output_dir = arguments['--output_dir']
    num_epochs = int(arguments['--epochs'])
    batch_size = int(arguments['--batch_size'])
    learning_rate = float(arguments['--lr'])

    print("Running regression task with the following arguments:")
    print(f"Input file: {input_file}")
    print(f"Embedding file: {embedding_file}")
    print(f"MI transformation: {mi_transformation}")
    print(f"Reg network: {reg_network}")
    print(f"Output directory: {output_dir}")
    print(f"Epochs: {num_epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    df = create_df(data_file=input_file, embs_file=embedding_file, mi_transformation=mi_transformation)

    X_train, X_tmp, y_train, y_tmp = train_test_split(df.iloc[:, :BERT_OUTPUT_SHAPE], df['mi_score'], random_state=42, test_size=0.3)
    X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, random_state=42, test_size=0.5)
    train_dataloader = create_data_loader(X_train, y_train, batch_size, shuffle=True)
    validation_dataloader = create_data_loader(X_val, y_val, batch_size, shuffle=False)
    test_dataloader = create_data_loader(X_test, y_test, batch_size, shuffle=False)

    model = BERT_Regressor(input_dim=BERT_OUTPUT_SHAPE, hidden_layers_config=reg_network)
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    results = []

    print('Start training...')

    for epoch in range(1, num_epochs + 1):
        start_time = time.time()
        model.train()
        total_loss = 0
        for ids, embeddings, targets in train_dataloader:
            outputs = model(embeddings)
            loss = criterion(outputs, targets)
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_train_loss = total_loss / len(train_dataloader)

        model.eval()
        val_total_loss = 0
        with torch.no_grad():
            for val_ids, val_embeddings, val_targets in validation_dataloader:
                val_outputs = model(val_embeddings)
                val_loss = criterion(val_outputs, val_targets)

                val_total_loss += val_loss.item()

        avg_val_loss = val_total_loss / len(validation_dataloader)
        epoch_time = time.time() - start_time

        result_entry = {'epoch': epoch,
                        'avg_train_loss': avg_train_loss,
                        'avg_val_loss': avg_val_loss,
                        'epoch_time': epoch_time}
        results.append(result_entry)
        print('\n'.join(key + ': ' + str(value) for key, value in result_entry.items()) + '\n')

    results_to_files(results_dict=results, output_dir=output_dir)

    print('Start testing...')

    model.eval()

    out_df = pd.DataFrame(columns=['target_mi', 'predicted_mi', 'abs_mi_err',
                                   'ent1', 'label1', 'ent2', 'label2', 'masked_sent'])
    test_total_loss = 0
    with torch.no_grad():
        for test_ids, test_embeddings, test_targets in test_dataloader:
            test_outputs = model(test_embeddings)

            # Calculate loss
            test_loss = criterion(test_outputs, test_targets)
            test_total_loss += test_loss.item()

            # log results
            absolute_errors = torch.abs(test_outputs - test_targets)
            batch_results = pd.DataFrame({'sent_ids': test_ids.squeeze().numpy(),
                                       'target_mi': test_targets.squeeze().numpy(),
                                       'predicted_mi': test_outputs.squeeze().numpy(),
                                       'abs_mi_err': absolute_errors.squeeze().numpy()})
            batch_results = batch_results.set_index('sent_ids', drop=True)
            batch_results.index.name = None # remove index column name

            batch_data = df.loc[torch.squeeze(test_ids)]
            batch_data = batch_data[['label1', 'label2', 'ent1', 'ent2', 'masked_sent']]

            batch_df = pd.concat([batch_results, batch_data], axis=1)

            out_df = out_df.append(batch_df, ignore_index=False)

    avg_test_loss = test_total_loss / len(test_dataloader)

    out_df.to_csv(f'{output_dir}/test_predictions_results.csv', index=True)
    with open(f'{output_dir}/test_report.txt', 'w') as file:
        file.write(f'Average test loss: {avg_test_loss}')