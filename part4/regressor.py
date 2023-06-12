import pandas as pd
import sys
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import BertTokenizerFast, AutoModel
from transformers.optimization import AdamW
import torch
from torch.utils.data import TensorDataset, DataLoader
import time
import warnings
warnings.filterwarnings("ignore", category=FutureWarning) # Disable the warning
import matplotlib.pyplot as plt
import torch.nn as nn

REGRESSION_NETWORK_HIDDEN_LAYERS_CONFIG = [512, None, 128, None]

if len(sys.argv) == 1:
    raise ValueError("Path to dataset missing")
else:
    input_file = sys.argv[1]

MI_TRANSFORMATION = 'ln' # can be either None, minmax, ln, or sqrt

BERT_MODEL = 'bert-base-cased'
BATCH_SIZE = 32

LEARNING_RATE = 1e-5
NUM_EPOCHS = 10

BERT_OUTPUT_SHAPE = 768

class BERT_Regressor(nn.Module):

    def __init__(self, input_dim, hidden_layers_config):
        '''
        :param input_dim: the input dimension of the network
        :param hidden_layers_config: indicates the hidden layers configuration of the network. \
                                     its format: [hidden_dim_1, dropout_rate_1, hidden_dim_2, dropout_rate_2, ...]. \
                                     for no dropout layer, use None value.
        '''
        super(BERT_Regressor, self).__init__()
        layers = self.create_model_layers(input_dim, hidden_layers_config, 1)
        self.model = nn.Sequential(*layers)

    def create_model_layers(self, input_dim, hidden_config_dims, output_dim):
        layers = []

        prev_dim = input_dim
        num_layers = len(hidden_config_dims) // 2  # Each layer has a dim and dropout rate

        for i in range(num_layers):
            dim = hidden_config_dims[i * 2]
            dropout_rate = hidden_config_dims[i * 2 + 1]

            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.ReLU())

            if dropout_rate is not None:
                layers.append(nn.Dropout(dropout_rate))

            prev_dim = dim

        layers.append(nn.Linear(prev_dim, output_dim))

        return layers

    def forward(self, x):
        return self.model(x)

####################

def transform_mi(series, transformation_type):
    if transformation_type is None:
        scaled_series = series
    else:
        scaler = MinMaxScaler(feature_range=(1, 100)) # we start all scaling with minmax scaling
        scaled_data = scaler.fit_transform(series.values.reshape(-1, 1))
        if transformation_type == 'minmax':
            pass
        elif transformation_type == 'sqrt':
            scaled_data = np.sqrt(scaled_data)
        elif transformation_type == 'ln':
            scaled_data = np.log(scaled_data)
        elif transformation_type == 'log10':
            scaled_data = np.log10(scaled_data)
        else:
            raise ValueError(f'Unknown MI transformation: {transformation_type}')
        scaled_series = pd.Series(scaled_data.flatten())
    return scaled_series

def create_data_loader(tokenizer, bert_model, X, y, max_length, batch_size): # TODO remove max_length
    embeddings_list = []
    y_list = []

    for i in range(0, len(X), batch_size): # we need to create the dataset in batches, otherwise bert will crash
        batch_X = X[i:i+batch_size]
        batch_y = y[i:i+batch_size]

        encoded_inputs = tokenizer(batch_X.tolist(), padding=True, truncation=True, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = bert_model(**encoded_inputs)

        embeddings = outputs.last_hidden_state[:, 0, :]
        y_tensor = torch.tensor(batch_y.values).unsqueeze(1)

        embeddings_list.append(embeddings)
        y_list.append(y_tensor)

    embeddings = torch.cat(embeddings_list, dim=0)
    y_tensor = torch.cat(y_list, dim=0)

    dataset = TensorDataset(embeddings, y_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prepare the data
    start_time = time.time()

    df = pd.read_csv(input_file)
    df['mi_score'] = df['mi_score'].astype('float32')
    df['mi_score'] = transform_mi(df['mi_score'], MI_TRANSFORMATION)
    max_length = max([len(s.split()) for s in df['masked_sent']])

    X_train, X_val, y_train, y_val = train_test_split(df['masked_sent'], df['mi_score'], random_state=42, test_size=0.3)

    bert_model = AutoModel.from_pretrained(BERT_MODEL).to(device)
    tokenizer = BertTokenizerFast.from_pretrained(BERT_MODEL)

    train_dataloader = create_data_loader(tokenizer, bert_model, X_train, y_train, max_length, BATCH_SIZE)
    validation_dataloader = create_data_loader(tokenizer, bert_model, X_val, y_val, max_length, BATCH_SIZE)

    print(f'Data preparation ended. it took: {time.time() - start_time} seconds')

    # Preparing the model
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
            embeddings = batch[0].to(device)
            targets = batch[1].to(device)

            optimizer.zero_grad()

            outputs = model(embeddings)
            loss = criterion(outputs, targets)
            total_loss += loss.item()

            loss.backward()
            optimizer.step()

        avg_train_loss = total_loss / len(train_dataloader)

        model.eval()
        val_total_loss = 0
        with torch.no_grad():
            for val_batch in validation_dataloader:
                val_embeddings = batch[0].to(device)
                val_targets = batch[1].to(device)

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

    results_df = pd.DataFrame(results).set_index('epoch')

    for measurement in results_df.columns:
        results_df[measurement].plot(title=measurement.replace('_', ' '))
        plt.savefig(f'{measurement}.jpg')
        plt.cla()

    results_df.to_csv('results.csv', index=True)