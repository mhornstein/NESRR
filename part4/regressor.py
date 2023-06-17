from sklearn.model_selection import train_test_split
from transformers.optimization import AdamW
import torch
from torch.utils.data import TensorDataset, DataLoader
import time
import warnings
warnings.filterwarnings("ignore", category=FutureWarning) # Disable the warning
import os
import sys

sys.path.append('../')
from common.regressor_util import *

REGRESSION_NETWORK_HIDDEN_LAYERS_CONFIG = [512, None, 128, None]

BATCH_SIZE = 32
LEARNING_RATE = 1e-5
NUM_EPOCHS = 10

MI_TRANSFORMATION = None  # can be either None, or: 'sqrt', 'ln', 'log10'

if len(sys.argv) < 3:
    raise ValueError("Path to dataset missing")
else:
    input_file = sys.argv[1]
    embeddings_file = sys.argv[2]

BERT_OUTPUT_SHAPE = 768

OUTPUT_DIR = 'results'

if not os.path.exists(f'./{OUTPUT_DIR}'):
    os.makedirs(f'{OUTPUT_DIR}')

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

    df = create_df(data_file=input_file, embs_file=embeddings_file, mi_transformation=MI_TRANSFORMATION)

    X_train, X_tmp, y_train, y_tmp = train_test_split(df.iloc[:, :BERT_OUTPUT_SHAPE], df['mi_score'], random_state=42, test_size=0.3)
    X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, random_state=42, test_size=0.5)
    train_dataloader = create_data_loader(X_train, y_train, BATCH_SIZE, shuffle=True)
    validation_dataloader = create_data_loader(X_val, y_val, BATCH_SIZE, shuffle=False)
    test_dataloader = create_data_loader(X_test, y_test, BATCH_SIZE, shuffle=False)

    model = BERT_Regressor(input_dim=BERT_OUTPUT_SHAPE, hidden_layers_config=REGRESSION_NETWORK_HIDDEN_LAYERS_CONFIG)
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    results = []

    print('Start training...')

    for epoch in range(1, NUM_EPOCHS + 1):
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

    results_to_files(results_dict=results, output_dir=OUTPUT_DIR)

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

    out_df.to_csv(f'{OUTPUT_DIR}/test_predictions_results.csv', index=True)
    with open(f'{OUTPUT_DIR}/test_report.txt', 'w') as file:
        file.write(f'Average test loss: {avg_test_loss}')