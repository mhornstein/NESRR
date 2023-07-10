from sklearn.model_selection import train_test_split
from transformers.optimization import AdamW
import torch
from torch.utils.data import TensorDataset, DataLoader
import time
import warnings
import os
warnings.filterwarnings("ignore", category=FutureWarning) # Disable the warning
import sys
from sklearn.metrics import classification_report
import random

sys.path.append('../')
from common.util import *
from common.classifier_util import *

BERT_OUTPUT_SHAPE = 768

class BERT_Classifier(nn.Module):

    def __init__(self, input_dim, hidden_layers_config):
        '''
        :param input_dim: the input dimension of the network
        :param hidden_layers_config: indicates the hidden layers configuration of the network. \
                                     its format: [hidden_dim_1, dropout_rate_1, hidden_dim_2, dropout_rate_2, ...]. \
                                     for no dropout layer, use None value.
        '''
        super(BERT_Classifier, self).__init__()
        self.model = create_network(input_dim=input_dim, hidden_layers_config=hidden_layers_config, output_dim=2)

    def forward(self, x):
        return self.model(x)

####################
def create_data_loader(X, y, batch_size, shuffle):
    sent_ids = torch.tensor(X.index, dtype=torch.int64).unsqueeze(dim=1).to(device)
    X_tensor = torch.tensor(X.values, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y, dtype=torch.int64).to(device)
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

def calc_measurements(model, dataloader, criterion):
    total_loss = total_good = total = 0
    with torch.no_grad():
        for sent_ids, embeddings, targets in dataloader:
            outputs = model(embeddings)
            total_loss += criterion(outputs, targets).item()
            predictions = logit_to_predicted_label(outputs)
            total_good += (predictions == targets).sum().item()
            total += len(targets)

    avg_loss = total_loss / total
    avg_acc = total_good / total

    return avg_loss, avg_acc

def run_experiment(df, score, score_threshold_type, score_threshold_value, hidden_layers_config, learning_rate, batch_size, num_epochs, output_dir):
    total_start_time = time.time()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Preparing the data
    X_train, X_tmp, y_train, y_tmp = train_test_split(df.iloc[:, :BERT_OUTPUT_SHAPE], df[score], random_state=42, test_size=0.3)
    y_train, y_tmp = score_to_label(y_train, y_tmp, score_threshold_type, score_threshold_value)

    X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, random_state=42, test_size=0.5)
    train_dataloader = create_data_loader(X_train, y_train, batch_size, shuffle=True)
    validation_dataloader = create_data_loader(X_val, y_val, batch_size, shuffle=False)
    test_dataloader = create_data_loader(X_test, y_test, batch_size, shuffle=False)

    # Preparing the model
    model = BERT_Classifier(input_dim=BERT_OUTPUT_SHAPE, hidden_layers_config=hidden_layers_config)
    model.to(device)

    # Preparing the loss: due to data imbalance, we will use weighted loss function instead of the out-of-the-box BERT's.
    # reference: https://discuss.huggingface.co/t/class-weights-for-bertforsequenceclassification/1674/6
    weight = calc_weight(y_train)
    criterion = nn.CrossEntropyLoss(weight=weight, reduction='mean')

    # Preparing the optimizer
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    # Initialize results with the random state of the model
    print('Evaluating beginning state state... ')
    model.eval()

    avg_train_loss, avg_train_acc = calc_measurements(model, train_dataloader, criterion)
    avg_val_loss, avg_val_acc = calc_measurements(model, validation_dataloader, criterion)

    result_entry = {'epoch': 0,
                    'avg_train_loss': avg_train_loss,
                    'avg_val_loss': avg_val_loss,
                    'avg_train_acc': avg_train_acc,
                    'avg_val_acc': avg_val_acc,
                    'epoch_time': 0}

    results = [result_entry]

    # start training. reference: https://huggingface.co/transformers/v3.2.0/custom_datasets.html
    print('Start training...')
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

        avg_train_loss, avg_train_acc = calc_measurements(model, train_dataloader, criterion)
        avg_val_loss, avg_val_acc = calc_measurements(model, validation_dataloader, criterion)

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
        for batch_i, (test_sent_ids, test_embeddings, test_targets) in enumerate(test_dataloader, start=1):
            print(f'Testing batch: {batch_i}/{len(test_dataloader)}')
            test_outputs = model(test_embeddings)
            test_total_loss += criterion(test_outputs, test_targets).item()

            test_predictions = logit_to_predicted_label(test_outputs)
            is_correct = test_targets == test_predictions

            all_test_targets += test_targets.tolist()
            all_test_predictions += test_predictions.tolist()

            batch_df = create_batch_result_df(data_df=df, sent_ids=test_sent_ids, targets=test_targets,
                                              predictions=test_predictions, is_correct=is_correct)

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

    networks_config_experiment_count = 5
    networks_sizes = [64, 128, 256, 512]

    exp_index = 1
    experiments_settings_list = []

    embeddings_file = '../data/dummy/embeddings_dummy.out' # '../data/embeddings.out'
    input_file = '../data/dummy/dummy_data.csv' # '../data/data.csv'

    input_df = create_df(input_file, embeddings_file)

    for score in ['mi_score', 'pmi_score']:
        for score_threshold_type in ['percentile', 'std_dist']:
            score_thresholds = [0.25, 0.5, 0.75] if score_threshold_type == 'percentile' else [-2, -1, 1, 2]
            for score_threshold_value in score_thresholds:
                for learning_rate in [0.01, 0.05, 0.001, 0.005]:
                    for batch_size in [64, 128, 256]:
                        for i in range(networks_config_experiment_count):
                            hidden_layers_config = random.sample(networks_sizes, random.randint(2, 4))
                            hidden_layers_config = [item for layer in hidden_layers_config for item in (layer, None)] # set dropout to be None for now
                            output_dir = f'{result_dir}/{exp_index}'
                            experiment_settings = {
                                                    'exp_index': exp_index,
                                                    'score': score,
                                                    'score_threshold_type': score_threshold_type,
                                                    'score_threshold_value': score_threshold_value,
                                                    'hidden_layers_config': hidden_layers_config,
                                                    'learning_rate': learning_rate,
                                                    'batch_size': batch_size,
                                                    'num_epochs': num_epochs
                                                   }
                            experiment_settings_str = ','.join([f'{key}={value}' for key, value in experiment_settings.items()])
                            print('running: ' + experiment_settings_str)
                            run_experiment(input_df, score, score_threshold_type, score_threshold_value,
                                           hidden_layers_config, learning_rate, batch_size, num_epochs, output_dir)
                            experiments_settings_list.append(experiment_settings)
                            exp_index += 1
    settings_df = pd.DataFrame(experiments_settings_list).set_index('exp_index')
    settings_df.to_csv(f"{result_dir}/experiments_settings.csv")