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
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

sys.path.append('../')
from common.regressor_util import *

BERT_MODEL = 'bert-base-cased'

####################

def create_data_loader(tokenizer, X, y, max_length, batch_size, shuffle):
    tokens = tokenizer.batch_encode_plus(X.tolist(), max_length=max_length, pad_to_max_length=True, truncation=True, return_tensors='pt')

    sent_ids = torch.tensor(X.index, dtype=torch.int64).unsqueeze(dim=1).to(device)
    ids = tokens['input_ids'].to(device)
    mask = tokens['attention_mask'].to(device)
    y_tensor = torch.tensor(y, dtype=torch.int64).to(device)

    dataset = TensorDataset(sent_ids, ids, mask, y_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return dataloader

def logit_to_predicted_label(logits):
    probs = logits.softmax(1)
    labels = probs.argmax(1)
    return labels

def score_to_label(y_train, y_tmp, score_threshold_type, score_threshold_value):
    train_description = y_train.describe()
    if score_threshold_type == 'percentile':
        precentile_key = f'{int(score_threshold_value * 100)}%'
        treshold = train_description[precentile_key]
    elif score_threshold_type == 'std_dist':
        mean, std = train_description['mean'], train_description['std']
        treshold = mean + score_threshold_value * std
    else:
        raise ValueError(f'Unknown score_threshold_type: {score_threshold_type}')

    y_train = np.where(y_train < treshold, 0, 1)
    y_tmp = np.where(y_tmp < treshold, 0, 1)

    return y_train, y_tmp

if __name__ == '__main__':
    total_start_time = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_file = '../data/dummy/dummy_data.csv'
    score = 'pmi_score' # can be either mi_score or pmi_score
    score_threshold_type = 'percentile' # can be either percentile or std_dist
    score_threshold_value = 0.75
    learning_rate=0.01
    batch_size=64
    num_epochs=1
    output_dir='results'

    print("input_file:", input_file)
    print("score:", score)
    print("score_threshold_type:", score_threshold_type)
    print("score_threshold_value:", score_threshold_value)
    print("learning_rate:", learning_rate)
    print("batch_size:", batch_size)
    print("num_epochs:", num_epochs)
    print("output_dir:", output_dir)
    print()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Preparing the data
    df = pd.read_csv(input_file).set_index('sent_id')
    X_train, X_tmp, y_train, y_tmp = train_test_split(df['masked_sent'], df[score], random_state=42, test_size=0.3)
    y_train, y_tmp = score_to_label(y_train, y_tmp, score_threshold_type, score_threshold_value)

    X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, random_state=42, test_size=0.5)

    tokenizer = BertTokenizerFast.from_pretrained(BERT_MODEL)
    max_length = max([len(s.split()) for s in df['masked_sent']])
    train_dataloader = create_data_loader(tokenizer, X_train, y_train, max_length, batch_size, shuffle=True)
    validation_dataloader = create_data_loader(tokenizer, X_val, y_val, max_length, batch_size, shuffle=False)
    test_dataloader = create_data_loader(tokenizer, X_test, y_test, max_length, batch_size, shuffle=False)

    # Preparing the model
    config = BertConfig.from_pretrained(BERT_MODEL, num_labels=2)  # Set num_labels=2 for classification
    model = BertForSequenceClassification(config)
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=learning_rate)

    # start training. reference: https://huggingface.co/transformers/v3.2.0/custom_datasets.html
    results = []
    print('Start training...')

    for epoch in range(1, num_epochs + 1):
        start_time = time.time()
        model.train()
        total_loss = 0
        total_acc = 0
        for sent_ids, input_ids, attention_mask, targets in train_dataloader:
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=targets)
            loss = outputs.loss
            total_loss += loss.item()
            predictions = logit_to_predicted_label(outputs.logits)
            accuracy = accuracy_score(targets.tolist(), predictions.tolist())
            total_acc += accuracy

            loss.backward()
            optimizer.step()

        avg_train_loss = total_loss / len(train_dataloader)
        avg_train_acc = total_acc / len(train_dataloader)

        model.eval()
        val_total_loss = 0
        val_total_acc = 0
        with torch.no_grad():
            for val_sent_ids, val_input_ids, val_attention_mask, val_targets in validation_dataloader:
                val_outputs = model(val_input_ids, attention_mask=val_attention_mask, labels=val_targets)
                val_total_loss += val_outputs.loss.item()
                val_predictions = logit_to_predicted_label(val_outputs.logits)
                val_accuracy = accuracy_score(val_targets.tolist(), val_predictions.tolist())
                val_total_acc += val_accuracy

        avg_val_loss = val_total_loss / len(validation_dataloader)
        avg_val_acc = val_total_acc / len(validation_dataloader)
        epoch_time = time.time() - start_time

        result_entry = {'epoch': epoch,
                        'avg_train_loss': avg_train_loss,
                        'avg_val_loss': avg_val_loss,
                        'avg_train_acc': avg_val_acc,
                        'avg_val_acc': avg_val_acc,
                        'epoch_time': epoch_time}
        results.append(result_entry)
        print('\n'.join(key + ': ' + str(value) for key, value in result_entry.items()) + '\n')

    results_to_files(results_dict=results, output_dir=output_dir)

    print('Start testing...')

    model.eval()
    all_test_targets = []
    all_test_predictions = []

    out_df = pd.DataFrame(columns=['target_label', 'predicted_label', 'is_correct',
                                   'ent1', 'label1', 'ent2', 'label2', 'masked_sent'])
    test_total_loss = 0
    with torch.no_grad():
        for test_sent_ids, test_input_ids, test_attention_mask, test_targets in test_dataloader:
            test_outputs = model(test_input_ids, attention_mask=test_attention_mask, labels=test_targets)
            test_predictions = logit_to_predicted_label(test_outputs.logits)
            is_correct = test_targets == test_predictions
            test_total_loss += test_outputs.loss.item()

            all_test_targets += test_targets.tolist()
            all_test_predictions += test_predictions.tolist()

            batch_results = pd.DataFrame({'sent_ids': test_sent_ids.squeeze().numpy(),
                                          'target_label': test_targets.squeeze().numpy(),
                                          'predicted_label': test_predictions.squeeze().numpy(),
                                          'is_correct': is_correct.squeeze().numpy()})
            batch_results = batch_results.set_index('sent_ids', drop=True)
            batch_results.index.name = None  # remove index column name

            batch_data = df.loc[torch.squeeze(test_sent_ids)]
            batch_data = batch_data[['label1', 'label2', 'ent1', 'ent2', 'masked_sent']]

            batch_df = pd.concat([batch_results, batch_data], axis=1)

            out_df = out_df.append(batch_df, ignore_index=False)

    avg_test_loss = test_total_loss / len(test_dataloader)

    test_classification_report = classification_report(all_test_targets, all_test_predictions, zero_division=1)

    total_time = time.time() - total_start_time
    print(f'Done. total time: {total_time} seconds')

    out_df.to_csv(f'{output_dir}/test_predictions_results.csv', index=True)
    with open(f'{output_dir}/report.txt', 'w') as file:
        file.write(f'Total time: {total_time}.\n\n')
        file.write(f'Test average loss: {avg_test_loss}.\n')
        file.write(f'Test classification report:\n')
        file.write(test_classification_report)

