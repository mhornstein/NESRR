"""
This script trains the BERT model for a regression task focused on predicting the mutual information score between two masked entities within a sentence.

Usage:
  regressor.py --input_file=<file> [--mi_trans=<mi_transformation>] [--output_dir=<directory>] [--epochs=<count>] [--batch_size=<size>] [--lr=<rate>]
  regressor.py (-h | --help)

Options:
  -h --help                    Show this screen.
  --input_file=<file>          Path to the dataset file.
  --mi_trans=<mi_transformation>  The transformation to apply for the mutual information scores.
                                 Available options: None, minmax, ln, sqrt. [default: None]
  --output_dir=<directory>     Path to the directory to write the results. [default: ./results]
  --epochs=<count>             Number of epochs for training. [default: 10]
  --batch_size=<size>          Batch size. [default: 32]
  --lr=<rate>                  Learning rate for training. [default: 1e-5]
"""

from sklearn.model_selection import train_test_split
from transformers import BertConfig, BertTokenizerFast, BertForSequenceClassification
from transformers.optimization import AdamW
import torch
from torch.utils.data import TensorDataset, DataLoader
import time
import warnings
import os
warnings.filterwarnings("ignore", category=FutureWarning) # Disable the warning
import sys
from docopt import docopt

sys.path.append('../')
from common.regressor_util import *

BERT_MODEL = 'bert-base-cased'

####################

def create_df(data_file, mi_transformation):
    df = pd.read_csv(data_file)
    df['mi_score'] = transform_mi(df['mi_score'], mi_transformation)
    df = df.set_index('sent_id')
    return df

def create_data_loader(tokenizer, X, y, max_length, batch_size, shuffle):
    tokens = tokenizer.batch_encode_plus(X.tolist(), max_length=max_length, pad_to_max_length=True, truncation=True, return_tensors='pt')

    sent_ids = torch.tensor(X.index, dtype=torch.int64).unsqueeze(dim=1).to(device)
    ids = tokens['input_ids'].to(device)
    mask = tokens['attention_mask'].to(device)
    y_tensor = torch.tensor(y.values.reshape(-1, 1), dtype=torch.float32).to(device)

    dataset = TensorDataset(sent_ids, ids, mask, y_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return dataloader

if __name__ == '__main__':
    total_start_time = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get command line arguments
    arguments = docopt(__doc__)

    input_file = arguments['--input_file']
    mi_transformation = arguments['--mi_trans']
    output_dir = arguments['--output_dir']
    num_epochs = int(arguments['--epochs'])
    batch_size = int(arguments['--batch_size'])
    learning_rate = float(arguments['--lr'])

    print("Running regression task with the following arguments:")
    print(f"Input file: {input_file}")
    print(f"MI transformation: {mi_transformation}")
    print(f"Output directory: {output_dir}")
    print(f"Epochs: {num_epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Preparing the data
    df = create_df(input_file, mi_transformation)
    X_train, X_tmp, y_train, y_tmp = train_test_split(df['masked_sent'], df['mi_score'], random_state=42, test_size=0.3)
    X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, random_state=42, test_size=0.5)

    tokenizer = BertTokenizerFast.from_pretrained(BERT_MODEL)
    max_length = max([len(s.split()) for s in df['masked_sent']])
    train_dataloader = create_data_loader(tokenizer, X_train, y_train, max_length, batch_size, shuffle=True)
    validation_dataloader = create_data_loader(tokenizer, X_val, y_val, max_length, batch_size, shuffle=False)
    test_dataloader = create_data_loader(tokenizer, X_test, y_test, max_length, batch_size, shuffle=False)

    # Preparing the model
    config = BertConfig.from_pretrained(BERT_MODEL, num_labels=1)  # Set num_labels=1 for regression
    model = BertForSequenceClassification(config)
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=learning_rate)

    results = []

    print('Start training...')

    for epoch in range(1, num_epochs + 1):
        start_time = time.time()
        model.train()
        total_loss = 0
        for sent_ids, input_ids, attention_mask, targets in train_dataloader:
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
            for val_sent_ids, val_input_ids, val_attention_mask, val_targets in validation_dataloader:
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

    results_to_files(results_dict=results, output_dir=output_dir)

    print('Start testing...')

    model.eval()

    out_df = pd.DataFrame(columns=['target_mi', 'predicted_mi', 'abs_mi_err',
                                   'ent1', 'label1', 'ent2', 'label2', 'masked_sent'])
    test_total_loss = 0
    with torch.no_grad():
        for test_sent_ids, test_input_ids, test_attention_mask, test_targets in test_dataloader:
            test_outputs = model(test_input_ids, attention_mask=test_attention_mask, labels=test_targets)
            absolute_errors = torch.abs(test_outputs.logits - test_targets)
            test_total_loss += test_outputs.loss.item()

            batch_results = pd.DataFrame({'sent_ids': test_sent_ids.squeeze().numpy(),
                                          'target_mi': test_targets.squeeze().numpy(),
                                          'predicted_mi': test_outputs.logits.squeeze().numpy(),
                                          'abs_mi_err': absolute_errors.squeeze().numpy()})
            batch_results = batch_results.set_index('sent_ids', drop=True)
            batch_results.index.name = None  # remove index column name

            batch_data = df.loc[torch.squeeze(test_sent_ids)]
            batch_data = batch_data[['label1', 'label2', 'ent1', 'ent2', 'masked_sent']]

            batch_df = pd.concat([batch_results, batch_data], axis=1)

            out_df = out_df.append(batch_df, ignore_index=False)

    avg_test_loss = test_total_loss / len(test_dataloader)

    out_df.to_csv(f'{output_dir}/test_predictions_results.csv', index=True)
    with open(f'{output_dir}/test_report.txt', 'w') as file:
        file.write(f'Average test loss: {avg_test_loss}')

    total_time = time.time() - total_start_time
    print(f'Done. total time: {total_time} seconds')