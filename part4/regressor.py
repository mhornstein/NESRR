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

MI_TRANSFORMATION = 'ln' # can be either None, minmax, ln, or sqrt

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

    df = df[['sent_id', 'mi_score']]
    df = df.set_index('sent_id')

    # Load embeddings file
    embs = pd.read_csv(embs_file, sep=' ', header=None)
    embs = embs.set_index(embs.columns[0])  # set sentence id as the index of the dataframe

    # combine both for a single df
    df = pd.concat([embs, df], axis=1)

    return df

def create_data_loader(X, y, batch_size):
    X_tensor = torch.tensor(X.values, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y.values.reshape(-1, 1), dtype=torch.float32).to(device)
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = create_df(data_file=input_file, embs_file=embeddings_file, mi_transformation=MI_TRANSFORMATION)

    X_train, X_val, y_train, y_val = train_test_split(df.iloc[:, :-1], df['mi_score'], random_state=42, test_size=0.3)
    train_dataloader = create_data_loader(X_train, y_train, BATCH_SIZE)
    validation_dataloader = create_data_loader(X_val, y_val, BATCH_SIZE)

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
        for batch in train_dataloader:
            embeddings = batch[0]
            targets = batch[1]

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
            for val_batch in validation_dataloader:
                val_embeddings = batch[0]
                val_targets = batch[1]

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