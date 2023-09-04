from sklearn.model_selection import train_test_split
from transformers.optimization import AdamW
from torch.utils.data import TensorDataset, DataLoader
import time
import warnings
warnings.filterwarnings("ignore", category=FutureWarning) # Disable the warning
import sys
import torch
from sklearn.metrics import classification_report, accuracy_score

sys.path.append('..\\')
from common.util import *

BERT_OUTPUT_SHAPE = 768

CONFIG_HEADER = ['exp_index', 'score', 'hidden_layers_config', 'learning_rate', 'batch_size', 'num_epochs']
RESULTS_HEADER = ['min_train_mse', 'min_train_mse_epoch', 'min_val_mse', 'min_val_mse_epoch', 'test_mse']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BERT_Regressor(nn.Module):

    def __init__(self, input_dim, hidden_layers_config):
        '''
        :param input_dim: the input dimension of the network
        :param hidden_layers_config: indicates the hidden layers configuration of the network. Its format: [hidden_dim_1, hidden_dim_2, ...]
        '''
        super(BERT_Regressor, self).__init__()
        self.model = create_network(input_dim=input_dim, hidden_layers_config=hidden_layers_config, output_dim=1)

    def forward(self, x):
        return self.model(x)

####################

def create_data_loader(X, y, batch_size, shuffle):
    sent_ids = torch.tensor(X.index, dtype=torch.int64).unsqueeze(dim=1).to(device)
    X_tensor = torch.tensor(X.values, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y.values, dtype=torch.float32).unsqueeze(dim=1).to(device)
    dataset = TensorDataset(sent_ids, X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader

def create_df(data_file, embs_file):
    df = pd.read_csv(data_file)
    df = df.set_index('sent_id')

    embs = pd.read_csv(embs_file, sep=' ', header=None)
    embs = embs.set_index(embs.columns[0])  # set sentence id as the index of the dataframe

    # combine both for a single df
    df = pd.concat([embs, df], axis=1)
    return df

def calc_loss(model, dataloader, criterion):
    total_loss = total = 0
    with torch.no_grad():
        for sent_ids, embeddings, targets in dataloader:
            outputs = model(embeddings)
            total_loss += criterion(outputs, targets).item()
            total += len(targets)

    avg_loss = total_loss / total
    return avg_loss

def results_to_files(results_df, output_dir):
    save_df_plot(df=results_df[['avg_train_loss', 'avg_val_loss']], title='mse', output_dir=output_dir)
    results_df.to_csv(f'{output_dir}/train_logs.csv', index=True)

def train_model(model, optimizer, num_epochs, train_dataloader, validation_dataloader, criterion, output_dir):
    print('Evaluating beginning state... ')
    model.eval()

    avg_train_loss = calc_loss(model, train_dataloader, criterion)
    avg_val_loss = calc_loss(model, validation_dataloader, criterion)

    result_entry = {'epoch': 0,
                    'avg_train_loss': avg_train_loss,
                    'avg_val_loss': avg_val_loss,
                    'epoch_time': 0}

    results = [result_entry]

    # start training. reference: https://huggingface.co/transformers/v3.2.0/custom_datasets.html
    for epoch in range(1, num_epochs + 1):
        print(f'Starting Epoch: {epoch}/{num_epochs}\n')
        start_time = time.time()
        model.train()
        for batch_i, (sent_ids, embeddings, targets) in enumerate(train_dataloader, start=1):
            print(f'Training batch: {batch_i}/{len(train_dataloader)}')
            outputs = model(embeddings)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('start evaluation ... ')
        model.eval()

        avg_train_loss = calc_loss(model, train_dataloader, criterion)
        avg_val_loss = calc_loss(model, validation_dataloader, criterion)

        epoch_time = time.time() - start_time

        result_entry = {'epoch': epoch,
                        'avg_train_loss': avg_train_loss,
                        'avg_val_loss': avg_val_loss,
                        'epoch_time': epoch_time}
        results.append(result_entry)
        print('Epoch report:')
        print('\n'.join(key + ': ' + str(value) for key, value in result_entry.items()) + '\n')

    results_df = pd.DataFrame(results).set_index('epoch')
    results_to_files(results_df=results_df, output_dir=output_dir)
    return results_df

def test_model(model, test_dataloader, df, criterion, output_dir):
    model.eval()

    out_df = pd.DataFrame(columns=['target_score', 'predicted_score', 'abs_error',
                                   'ent1', 'label1', 'ent2', 'label2', 'masked_sent'])
    test_total_loss = 0
    with torch.no_grad():
        for batch_i, (test_sent_ids, test_embeddings, test_targets) in enumerate(test_dataloader, start=1):
            print(f'Testing batch: {batch_i}/{len(test_dataloader)}')
            test_outputs = model(test_embeddings)
            abs_error = torch.abs(test_targets - test_outputs)
            loss = criterion(test_outputs, test_targets).item()
            test_total_loss += loss

            sent_ids = test_sent_ids.squeeze().numpy()

            batch_results = pd.DataFrame({'sent_ids': sent_ids,
                                          'target_score': test_targets.squeeze().numpy(),
                                          'predicted_score': test_outputs.squeeze().numpy(),
                                          'abs_error': abs_error.squeeze().numpy()})
            batch_results = batch_results.set_index('sent_ids', drop=True)
            batch_results.index.name = None  # remove index column name

            batch_data = df.loc[sent_ids]
            batch_data = batch_data[['label1', 'label2', 'ent1', 'ent2', 'masked_sent']]

            batch_df = pd.concat([batch_results, batch_data], axis=1)

            out_df = pd.concat([out_df, batch_df], ignore_index=False)

    out_df.to_csv(f'{output_dir}\\test_predictions_results.csv', index=True)

    test_size = len(out_df)
    avg_test_loss = test_total_loss / test_size

    return avg_test_loss

def run_experiment(df, score, hidden_layers_config, learning_rate, batch_size, num_epochs, output_dir):
    total_start_time = time.time()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Preparing the data
    X_train, X_tmp, y_train, y_tmp = train_test_split(df.iloc[:, :BERT_OUTPUT_SHAPE], df[score], random_state=42, test_size=0.4)
    X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, random_state=42, test_size=0.5)

    train_dataloader = create_data_loader(X_train, y_train, batch_size, shuffle=True)
    validation_dataloader = create_data_loader(X_val, y_val, batch_size, shuffle=False)
    test_dataloader = create_data_loader(X_test, y_test, batch_size, shuffle=False)

    # Preparing the model
    model = BERT_Regressor(input_dim=BERT_OUTPUT_SHAPE, hidden_layers_config=hidden_layers_config)
    model.to(device)

    criterion = nn.MSELoss()

    # Preparing the optimizer
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    # train model
    print('Start training...')
    train_results = train_model(model, optimizer, num_epochs, train_dataloader, validation_dataloader, criterion, output_dir)

    # test model
    print('Start testing...')
    test_loss = test_model(model, test_dataloader, df, criterion, output_dir)

    total_time = time.time() - total_start_time
    print(f'Done. total time: {total_time} seconds.\n')

    with open(f'{output_dir}\\total_time.txt', 'a') as file:
        file.write(f'Total time: {total_time} seconds.')

    experiment_results = {'min_train_mse': train_results['avg_train_loss'].min(),
                          'min_train_mse_epoch': train_results['avg_train_loss'].idxmin(),
                          'min_val_mse': train_results['avg_val_loss'].min(),
                          'min_val_mse_epoch': train_results['avg_val_loss'].idxmin(),
                          'test_mse': test_loss}
    return experiment_results

if __name__ == '__main__':
    if len(sys.argv) < 4:
        raise ValueError(f'Not enough arguments. Arguments given: {sys.argv[1:]}')

    input_file = sys.argv[1] # e.g. '../data/dummy/dummy_data.csv' or '../data/data.csv'
    embeddings_file = sys.argv[2] # e.g. '../data/dummy/embeddings_dummy.out' or '../data/embeddings.out'
    result_dir = sys.argv[3] # e.g. 'results'

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    experiment_log_file_path = f'{result_dir}\\experiments_logs.csv'
    exp_index = init_experiment_log_file(experiment_log_file_path, CONFIG_HEADER, RESULTS_HEADER)

    num_epochs = 40

    networks_config_experiment_count = 3

    input_df = create_df(input_file, embeddings_file)

    for score in ['mi_score', 'pmi_score']:
        for learning_rate in [0.01, 0.05, 0.001, 0.005]:
            for batch_size in [256, 128, 64]:
                for i in range(networks_config_experiment_count):
                    hidden_layers_config = draw_hidden_layers_config()
                    output_dir = f'{result_dir}\\{exp_index}'
                    experiment_settings = {
                                            'exp_index': exp_index,
                                            'score': score,
                                            'hidden_layers_config': hidden_layers_config,
                                            'learning_rate': learning_rate,
                                            'batch_size': batch_size,
                                            'num_epochs': num_epochs
                                           }
                    experiment_settings_str = ','.join([f'{key}={value}' for key, value in experiment_settings.items()])
                    print('running: ' + experiment_settings_str)
                    experiment_results = run_experiment(input_df, score, hidden_layers_config, learning_rate, batch_size, num_epochs, output_dir)
                    log_experiment(experiment_log_file_path, CONFIG_HEADER, experiment_settings, RESULTS_HEADER, experiment_results)
                    exp_index += 1