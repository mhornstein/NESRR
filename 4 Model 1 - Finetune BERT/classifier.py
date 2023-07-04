from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers.optimization import AdamW
import torch
from torch.utils.data import TensorDataset, DataLoader
import time
import warnings
import os
warnings.filterwarnings("ignore", category=FutureWarning) # Disable the warning
import sys
from sklearn.metrics import classification_report
import numpy as np

sys.path.append('../')
from common.util import *

BERT_MODEL = 'bert-base-uncased' #  'distilbert-base-uncased'

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
    labels = logits.argmax(1)
    return labels

def calc_weight(labels):
    '''
    Returns a tensor representing the weights for label 0 and label 1 respectively.
    for the weights, We want the positive class weight to be inversely proportional to the proportion of positive examples.
    If there are fewer positive examples, we increase the weight to give more importance to those examples during the loss computation.
    We use 2 for sclaling.
    Therefore, if N = #all-samples, P = #positive-samples: 1/ 2 * frequency = 1 / 2 * (P/N) = N / 2 * P
    The same apply for negative samples.
    Partial reference: https://forums.fast.ai/t/about-weighted-bceloss/78570/3
    '''
    total_examples = len(labels)
    num_positive = np.sum(labels == 1)
    num_negative = total_examples - num_positive

    positive_weight = total_examples / (2 * num_positive)
    negative_weight = total_examples / (2 * num_negative)
    return torch.tensor([negative_weight, positive_weight], dtype=torch.float32) # dtype of float 32 is the requirement of the weight

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

def run_experiment(input_file, score, score_threshold_type, score_threshold_value, learning_rate, batch_size, num_epochs, output_dir):
    print("input_file:", input_file)
    print("score:", score)
    print("score_threshold_type:", score_threshold_type)
    print("score_threshold_value:", score_threshold_value)
    print("learning_rate:", learning_rate)
    print("batch_size:", batch_size)
    print("num_epochs:", num_epochs)
    print("output_dir:", output_dir)
    print()

    total_start_time = time.time()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Preparing the data
    df = pd.read_csv(input_file).set_index('sent_id')
    X_train, X_tmp, y_train, y_tmp = train_test_split(df['masked_sent'], df[score], random_state=42, test_size=0.3)
    y_train, y_tmp = score_to_label(y_train, y_tmp, score_threshold_type, score_threshold_value)

    X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, random_state=42, test_size=0.5)

    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)
    max_length = max([len(s.split()) for s in df['masked_sent']])
    train_dataloader = create_data_loader(tokenizer, X_train, y_train, max_length, batch_size, shuffle=True)
    validation_dataloader = create_data_loader(tokenizer, X_val, y_val, max_length, batch_size, shuffle=False)
    test_dataloader = create_data_loader(tokenizer, X_test, y_test, max_length, batch_size, shuffle=False)

    # Preparing the model
    model = AutoModelForSequenceClassification.from_pretrained(BERT_MODEL, num_labels=2)
    model.to(device)

    # Preparing the loss: due to data imbalance, we will use weighted loss function instead of the out-of-the-box BERT's.
    # reference: https://discuss.huggingface.co/t/class-weights-for-bertforsequenceclassification/1674/6
    weight = calc_weight(y_train)
    criterion = nn.CrossEntropyLoss(weight=weight, reduction='mean')

    # Preparing the optimizer
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    # start training. reference: https://huggingface.co/transformers/v3.2.0/custom_datasets.html
    results = []

    for epoch in range(1, num_epochs + 1):
        print(f'Starting Epoch: {epoch}/{num_epochs}\n')
        start_time = time.time()
        model.train()
        total_loss = 0
        total_good = 0
        for batch_i, (sent_ids, input_ids, attention_mask, targets) in enumerate(train_dataloader, start=1):
            print(f'Training batch: {batch_i}/{len(train_dataloader)}')
            outputs = model(input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, targets)
            total_loss += loss.item()
            predictions = logit_to_predicted_label(outputs.logits)
            total_good += (predictions == targets).sum().item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total = len(X_train)
        avg_train_loss = total_loss / total
        avg_train_acc = total_good / total

        model.eval()
        val_total_loss = 0
        total_val_good = 0
        with torch.no_grad():
            for batch_i, (val_sent_ids, val_input_ids, val_attention_mask, val_targets) in enumerate(validation_dataloader, start=1):
                print(f'Validation batch: {batch_i}/{len(validation_dataloader)}')
                val_outputs = model(val_input_ids, attention_mask=val_attention_mask)
                val_total_loss += criterion(val_outputs.logits, val_targets).item()
                val_predictions = logit_to_predicted_label(val_outputs.logits)
                total_val_good += (val_predictions == val_targets).sum().item()

        total = len(X_val)
        avg_val_loss = val_total_loss / total
        avg_val_acc = total_val_good / total

        epoch_time = time.time() - start_time

        result_entry = {'epoch': epoch,
                        'avg_train_loss': avg_train_loss,
                        'avg_val_loss': avg_val_loss,
                        'avg_train_acc': avg_train_acc,
                        'avg_val_acc': avg_val_acc,
                        'epoch_time': epoch_time}
        results.append(result_entry)
        print('Epoch report:')
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
        for batch_i, (test_sent_ids, test_input_ids, test_attention_mask, test_targets) in enumerate(test_dataloader, start=1):
            print(f'Testing batch: {batch_i}/{len(test_dataloader)}')
            test_outputs = model(test_input_ids, attention_mask=test_attention_mask)
            test_total_loss += criterion(test_outputs.logits, test_targets).item()
            
            test_predictions = logit_to_predicted_label(test_outputs.logits)
            is_correct = test_targets == test_predictions

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

    avg_test_loss = test_total_loss / len(X_test)

    test_classification_report = classification_report(all_test_targets, all_test_predictions, zero_division=1)

    total_time = time.time() - total_start_time
    print(f'Done. total time: {total_time} seconds.\n')

    out_df.to_csv(f'{output_dir}/test_predictions_results.csv', index=True)

    experiment_settings = {
        'score': score,
        'score_threshold_type': score_threshold_type,
        'score_threshold_value': score_threshold_value,
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'num_epochs': num_epochs
    }
    with open(f'{output_dir}/report.txt', 'w') as file:
        file.write(f'Total time: {total_time}.\n\n')
        file.write(f'Test average loss: {avg_test_loss}.\n')
        file.write(f'Test classification report:\n')
        file.write(test_classification_report)
        file.write(f'Settings:\n{experiment_settings}')

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result_dir = 'results'

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    num_epochs = 10

    exp_index = 1
    experiments_settings_list = []

    input_file = '../data/data.csv'

    for score in ['mi_score', 'pmi_score']:
        for score_threshold_type in ['percentile', 'std_dist']:
            score_thresholds = [0.25, 0.5, 0.75] if score_threshold_type == 'percentile' else [-2, -1, 1, 2]
            for score_threshold_value in score_thresholds:
                for learning_rate in [0.01, 0.05, 0.001, 0.005]:
                    for batch_size in [64, 128, 256]:
                        output_dir = f'{result_dir}/{exp_index}'
                        run_experiment(input_file, score, score_threshold_type, score_threshold_value, learning_rate,
                                       batch_size, num_epochs, output_dir)
                        experiment_settings = {
                                                'exp_index': exp_index,
                                                'score': score,
                                                'score_threshold_type': score_threshold_type,
                                                'score_threshold_value': score_threshold_value,
                                                'learning_rate': learning_rate,
                                                'batch_size': batch_size,
                                                'num_epochs': num_epochs
                                               }
                        experiments_settings_list.append(experiment_settings)
                        exp_index += 1
    settings_df = pd.DataFrame(experiments_settings_list).set_index('exp_index')
    settings_df.to_csv(f"{result_dir}/experiments_settings.csv")



