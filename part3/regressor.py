import pandas as pd
import sys
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import BertConfig, BertTokenizerFast, BertForSequenceClassification
from transformers.optimization import AdamW
import torch
from torch.utils.data import TensorDataset, DataLoader
import time
import warnings
import os
warnings.filterwarnings("ignore", category=FutureWarning) # Disable the warning
import matplotlib.pyplot as plt

BATCH_SIZE = 32
LEARNING_RATE = 1e-5
NUM_EPOCHS = 10

MI_TRANSFORMATION = 'ln' # can be either None, minmax, ln, or sqrt

if len(sys.argv) == 1:
    raise ValueError("Path to dataset missing")
else:
    input_file = sys.argv[1]

BERT_MODEL = 'bert-base-cased'

OUTPUT_DIR = 'results'

if not os.path.exists(f'./{OUTPUT_DIR}'):
    os.makedirs(f'{OUTPUT_DIR}')

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

def create_df(data_file, mi_transformation):
    df = pd.read_csv(data_file)
    df['mi_score'] = transform_mi(df['mi_score'], mi_transformation)
    return df

def create_data_loader(tokenizer, X, y, max_length, batch_size):
    tokens = tokenizer.batch_encode_plus(X.tolist(), max_length=max_length, pad_to_max_length=True, truncation=True, return_tensors='pt')

    ids = tokens['input_ids']
    mask = tokens['attention_mask']
    y_tensor = torch.tensor(y.values.reshape(-1, 1), dtype=torch.float32).to(device)

    dataset = TensorDataset(ids, mask, y_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Preparing the data
    df = create_df(input_file, MI_TRANSFORMATION)
    X_train, X_val, y_train, y_val = train_test_split(df['masked_sent'], df['mi_score'], random_state=42, test_size=0.3)

    tokenizer = BertTokenizerFast.from_pretrained(BERT_MODEL)
    max_length = max([len(s.split()) for s in df['masked_sent']])
    train_dataloader = create_data_loader(tokenizer, X_train, y_train, max_length, BATCH_SIZE)
    validation_dataloader = create_data_loader(tokenizer, X_val, y_val, max_length, BATCH_SIZE)

    # Preparing the model
    config = BertConfig.from_pretrained(BERT_MODEL, num_labels=1)  # Set num_labels=1 for regression
    model = BertForSequenceClassification(config)
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    results = []

    for epoch in range(1, NUM_EPOCHS + 1):
        start_time = time.time()
        model.train()
        total_loss = 0
        for batch in train_dataloader:
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            targets = batch[2].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=targets)
            loss = outputs.loss
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_train_loss = total_loss / len(train_dataloader)

        model.eval()
        val_total_loss = 0
        with torch.no_grad():
            for val_batch in validation_dataloader:
                val_input_ids = val_batch[0].to(device)
                val_attention_mask = val_batch[1].to(device)
                val_targets = val_batch[2].to(device)

                val_outputs = model(val_input_ids, attention_mask=val_attention_mask, labels=val_targets)
                val_total_loss += val_outputs.loss.item()

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