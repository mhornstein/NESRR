import pandas as pd
from config import *
import sys
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import AutoModel, BertTokenizerFast
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
import warnings
warnings.filterwarnings("ignore", category=FutureWarning) # Disable the warning

if len(sys.argv) == 1:
    raise ValueError("Path to dataset missing")
else:
    input_file = sys.argv[1]

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

def create_data_loader(tokenizer, X, y, max_length, batch_size):
    tokens = tokenizer.batch_encode_plus(X.tolist(), max_length=max_length, pad_to_max_length=True, truncation=True, return_token_type_ids=False)
    seq = torch.tensor(tokens['input_ids'])
    mask = torch.tensor(tokens['attention_mask'])

    y_tensor = torch.tensor(y.tolist())

    dataset = TensorDataset(seq, mask, y_tensor)
    sampler = RandomSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)

    return dataloader

if __name__ == '__main__':
    df = pd.read_csv(input_file)
    df['mi_score'] = transform_mi(df['mi_score'], MI_TRANSFORMATION)
    max_length = max([len(s.split()) for s in df['masked_sent']])

    X_train, X_tmp, y_train, y_tmp = train_test_split(df['masked_sent'], df['mi_score'], random_state=42, test_size=0.3)
    X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, random_state=42, test_size=0.5)

    tokenizer = BertTokenizerFast.from_pretrained(BERT_MODEL)
    train_dataloader = create_data_loader(tokenizer, X_train, y_train, max_length, BATCH_SIZE)
    validation_dataloader = create_data_loader(tokenizer, X_val, y_val, max_length, BATCH_SIZE)

