import pandas as pd
import sys
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.model_selection import train_test_split
from transformers.optimization import AdamW
import torch
from torch.utils.data import TensorDataset, DataLoader
import time
import warnings
warnings.filterwarnings("ignore", category=FutureWarning) # Disable the warning
import matplotlib.pyplot as plt
import torch.nn as nn
import os
from sklearn.preprocessing import LabelEncoder

BATCH_SIZE = 32
LEARNING_RATE = 1e-5
NUM_EPOCHS = 10

MI_TRANSFORMATION = None # can be either None, minmax, ln, or sqrt

if len(sys.argv) < 3:
    raise ValueError("Path to dataset missing")
else:
    input_file = sys.argv[1]
    embeddings_file = sys.argv[2]

BERT_OUTPUT_SHAPE = 768

OUTPUT_DIR = 'results'

if not os.path.exists(f'./{OUTPUT_DIR}'):
    os.makedirs(f'{OUTPUT_DIR}')

LABEL_ENCODER = LabelEncoder()

####################

class BERT_Regressor(nn.Module):

    def __init__(self, input_dim, num_labels1, num_labels2):
        super(BERT_Regressor, self).__init__()
        self.input_dim = input_dim
        self.num_labels1 = num_labels1
        self.num_labels2 = num_labels2
        num_labels = self.num_labels1 + self.num_labels2

        # classification layer
        self.fc1 = nn.Linear(self.input_dim, 64)
        self.fc2 = nn.Linear(64, num_labels)

        # regression layer
        self.regression_layer = nn.Linear(num_labels + self.input_dim, 1)

    def forward(self, embs):
        # Step 1: classification
        x = self.fc1(embs)
        x = torch.relu(x)
        x = self.fc2(x)

        label1_classification_output = x[:, :self.num_labels1]
        label2_classification_output = x[:, self.num_labels1:]

        # Step 2: regression with the classification output
        combined_input = torch.cat((label1_classification_output, label2_classification_output, embs), dim=1)
        regression_output = self.regression_layer(combined_input)

        # Step 3: return the results
        return label1_classification_output, label2_classification_output, regression_output

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

def encode_column(df, column_name, encoder):
    '''
    This function converts the labels in a specified column of a DataFrame to ordinal numbers.
    The categorical labels are transformed into sequential integers representing an ordinal positions.
    :param df: the required dataframe
    :param column_name: the column for which the labels
    '''
    encoder.fit(df[column_name])
    df[column_name] = encoder.transform(df[column_name])

def create_df(data_file, embs_file, mi_transformation):
    # Load input data file
    df = pd.read_csv(data_file)
    df['mi_score'] = transform_mi(df['mi_score'], mi_transformation)

    df = df[['sent_id', 'label1', 'label2', 'mi_score']]
    encode_column(df, 'label1', LABEL_ENCODER)
    encode_column(df, 'label2', LABEL_ENCODER)
    df = df.set_index('sent_id')

    # Load embeddings file
    embs = pd.read_csv(embs_file, sep=' ', header=None)
    embs = embs.set_index(embs.columns[0])  # set sentence id as the index of the dataframe

    # combine both for a single df
    df = pd.concat([embs, df], axis=1)

    df = df[[col for col in df.columns if col != 'mi_score'] + ['mi_score']] # move mi to be the last column

    return df

def create_data_loader(X, y, batch_size):
    embs_tensor = torch.tensor(X.iloc[:, :768].values, dtype=torch.float32).to(device)
    label1_tensor = torch.tensor(X['label1'].values, dtype=torch.long).to(device) # Note that target class must be of type torch.long
    label2_tensor = torch.tensor(X['label2'].values, dtype=torch.long).to(device)
    mi_tensor = torch.tensor(y.values.reshape(-1, 1), dtype=torch.float32).to(device)
    dataset = TensorDataset(embs_tensor, label1_tensor, label2_tensor, mi_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = create_df(data_file=input_file, embs_file=embeddings_file, mi_transformation=MI_TRANSFORMATION)

    X_train, X_val, y_train, y_val = train_test_split(df.iloc[:, :-1], df['mi_score'], random_state=42, test_size=0.3)
    train_dataloader = create_data_loader(X_train, y_train, BATCH_SIZE)
    validation_dataloader = create_data_loader(X_val, y_val, BATCH_SIZE)

    label1_values = set(df['label1'].unique())
    label2_values = set(df['label2'].unique())
    model = BERT_Regressor(input_dim=BERT_OUTPUT_SHAPE, num_labels1=len(label1_values), num_labels2=len(label2_values))
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    classification_criterion = nn.CrossEntropyLoss()
    regression_criterion = nn.MSELoss()

    results = []

    print('Start training...')

    for epoch in range(1, NUM_EPOCHS + 1):
        start_time = time.time()
        model.train()
        total_loss = 0
        for batch in train_dataloader:
            embeddings, labels1, labels2, mi_score = batch

            label1_classification_output, label2_classification_output, regression_output = model(embeddings)

            label1_loss = classification_criterion(label1_classification_output, labels1)
            label2_loss = classification_criterion(label2_classification_output, labels2)
            regression_loss = regression_criterion(regression_output, mi_score)
            loss = label1_loss + label2_loss + regression_loss

            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_train_loss = total_loss / len(train_dataloader)

        model.eval()
        val_total_loss = 0
        with torch.no_grad():
            for val_batch in validation_dataloader:
                val_embeddings, val_labels1, val_labels2, val_mi_score = batch

                val_label1_classification_output, val_label2_classification_output, val_regression_output = model(val_embeddings)

                val_label1_loss = classification_criterion(val_label1_classification_output, val_labels1)
                val_label2_loss = classification_criterion(val_label2_classification_output, val_labels2)
                val_regression_loss = regression_criterion(val_regression_output, val_mi_score)
                val_loss = val_label1_loss + val_label2_loss + val_regression_loss

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
        plt.savefig(f'{OUTPUT_DIR}/{measurement}.jpg')
        plt.cla()

    results_df.to_csv(f'{OUTPUT_DIR}/results.csv', index=True)