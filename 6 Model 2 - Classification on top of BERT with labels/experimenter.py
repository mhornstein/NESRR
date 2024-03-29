from sklearn.model_selection import train_test_split
from transformers.optimization import AdamW
from torch.utils.data import TensorDataset, DataLoader
import time
import warnings
warnings.filterwarnings("ignore", category=FutureWarning) # Disable the warning
import sys
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

sys.path.append('..\\')
from common.util import *
from common.classifier_util import *

BERT_OUTPUT_SHAPE = 768

CONFIG_HEADER = ['exp_index', 'score', 'labels_pred_hidden_layers_config', 'interest_pred_hidden_layers_config' ,'learning_rate', 'batch_size', 'num_epochs']
RESULTS_HEADER = ['max_train_acc', 'max_train_acc_epoch', 'max_train_labels_acc', 'max_train_labels_acc_epoch',
                  'max_val_acc', 'max_val_acc_epoch', 'max_val_labels_acc', 'max_val_labels_acc_epoch',
                  'test_acc']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BERT_Classifier(nn.Module):

    def __init__(self, input_dim, labels_pred_hidden_layers_config, interest_pred_hidden_layers_config, num_labels):
        super(BERT_Classifier, self).__init__()
        self.num_labels = num_labels
        self.labels_network = create_network(input_dim=input_dim,
                                             hidden_layers_config=labels_pred_hidden_layers_config,
                                             output_dim=num_labels * 2)
        self.interest_network = create_network(input_dim=num_labels * 2 + input_dim,
                                             hidden_layers_config=interest_pred_hidden_layers_config,
                                             output_dim=2)

    def forward(self, embs):
        # Step 1: classify the labels
        x = self.labels_network(embs)

        label1_classification_output = x[:, :self.num_labels]
        label2_classification_output = x[:, self.num_labels:]

        # Step 2: inteset classification with the labels classification output
        combined_input = torch.cat((label1_classification_output, label2_classification_output, embs), dim=1)
        interest_classification_output = self.interest_network(combined_input)

        # Step 3: return the results
        return label1_classification_output, label2_classification_output, interest_classification_output

####################

def create_data_loader(X, y, batch_size, shuffle):
    ids_tensor = torch.tensor(X.index, dtype=torch.int64).unsqueeze(dim=1).to(device)
    embs_tensor = torch.tensor(X.iloc[:, :BERT_OUTPUT_SHAPE].values, dtype=torch.float32).to(device)
    label1_tensor = torch.tensor(X['label1'].values, dtype=torch.long).to(device) # Note that target class must be of type torch.long
    label2_tensor = torch.tensor(X['label2'].values, dtype=torch.long).to(device)
    y_tensor = torch.tensor(y, dtype=torch.int64).to(device)
    dataset = TensorDataset(ids_tensor, embs_tensor, label1_tensor, label2_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader

def create_df(data_file, embs_file):
    df = pd.read_csv(data_file)
    df = df.set_index('sent_id')

    # Add the embeddings
    embs = pd.read_csv(embs_file, sep=' ', header=None)
    embs = embs.set_index(embs.columns[0])  # set sentence id as the index of the dataframe
    df = pd.concat([embs, df], axis=1)

    return df

def calc_measurements(model, dataloader, interest_criterion, labels_criterion):
    total_loss = interest_total_good = labels_total_good = total = 0
    with torch.no_grad():
        for sent_ids, embeddings, labels1, labels2, targets in dataloader:
            label1_classification_output, label2_classification_output, interest_classification_output = model(embeddings)
            label1_loss = labels_criterion(label1_classification_output, labels1)
            label2_loss = labels_criterion(label2_classification_output, labels2)
            interest_loss = interest_criterion(interest_classification_output, targets)
            loss = label1_loss + label2_loss + interest_loss
            total_loss += loss.item()

            predictions = logit_to_predicted_label(interest_classification_output)
            interest_total_good += (predictions == targets).sum().item()

            predictions = logit_to_predicted_label(label1_classification_output)
            labels_total_good += (predictions == labels1).sum().item()

            predictions = logit_to_predicted_label(label2_classification_output)
            labels_total_good += (predictions == labels2).sum().item()

            total += len(targets)

    avg_loss = total_loss / total
    avg_interest_acc = interest_total_good / total
    avg_labels_acc = labels_total_good / (2 * total)

    return avg_loss, avg_labels_acc, avg_interest_acc

def train_model(model, optimizer, num_epochs, train_dataloader, validation_dataloader, interest_criterion, labels_criterion, output_dir):
    print('Evaluating beginning state... ')
    model.eval()

    avg_train_loss, avg_train_labels_acc, avg_train_interest_acc = calc_measurements(model, train_dataloader, interest_criterion, labels_criterion)
    avg_val_loss, avg_val_labels_acc, avg_val_interest_acc = calc_measurements(model, validation_dataloader, interest_criterion, labels_criterion)

    result_entry = {'epoch': 0,
                    'avg_train_loss': avg_train_loss,
                    'avg_val_loss': avg_val_loss,
                    'avg_train_labels_acc': avg_train_labels_acc,
                    'avg_val_labels_acc': avg_val_labels_acc,
                    'avg_train_acc': avg_train_interest_acc,
                    'avg_val_acc': avg_val_interest_acc,
                    'epoch_time': 0}

    results = [result_entry]

    # start training. reference: https://huggingface.co/transformers/v3.2.0/custom_datasets.html
    for epoch in range(1, num_epochs + 1):
        print(f'Starting Epoch: {epoch}/{num_epochs}\n')
        start_time = time.time()
        model.train()
        for batch_i, (sent_ids, embeddings, labels1, labels2, targets) in enumerate(train_dataloader, start=1):
            print(f'Training batch: {batch_i}/{len(train_dataloader)}')
            label1_classification_output, label2_classification_output, interest_classification_output = model(embeddings)

            label1_loss = labels_criterion(label1_classification_output, labels1)
            label2_loss = labels_criterion(label2_classification_output, labels2)
            interest_loss = interest_criterion(interest_classification_output, targets)
            loss = label1_loss + label2_loss + interest_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('start evaluation ... ')
        model.eval()

        avg_train_loss, avg_train_labels_acc, avg_train_interest_acc = calc_measurements(model, train_dataloader, interest_criterion, labels_criterion)
        avg_val_loss, avg_val_labels_acc, avg_val_interest_acc = calc_measurements(model, validation_dataloader, interest_criterion, labels_criterion)

        epoch_time = time.time() - start_time

        result_entry = {'epoch': epoch,
                        'avg_train_loss': avg_train_loss,
                        'avg_val_loss': avg_val_loss,
                        'avg_train_labels_acc': avg_train_labels_acc,
                        'avg_val_labels_acc': avg_val_labels_acc,
                        'avg_train_acc': avg_train_interest_acc,
                        'avg_val_acc': avg_val_interest_acc,
                        'epoch_time': epoch_time}
        results.append(result_entry)
        print('Epoch report:')
        print('\n'.join(key + ': ' + str(value) for key, value in result_entry.items()) + '\n')

    results_df = pd.DataFrame(results).set_index('epoch')
    results_to_files(results_df=results_df, output_dir=output_dir)
    return results_df

def test_model(model, test_dataloader, df, interest_criterion, labels_criterion, le, output_dir):
    model.eval()
    all_test_targets = []
    all_test_predictions = []

    out_df = pd.DataFrame(columns=['target_label', 'predicted_label', 'is_correct',
                                   'label1_predictions', 'label2_predictions',
                                   'ent1', 'label1', 'ent2', 'label2', 'masked_sent'])
    with torch.no_grad():
        for batch_i, (sent_ids, test_embeddings, test_labels1, test_labels2, test_targets) in enumerate(test_dataloader, start=1):
            print(f'Testing batch: {batch_i}/{len(test_dataloader)}')
            label1_classification_output, label2_classification_output, interest_classification_output = model(test_embeddings)

            test_predictions = logit_to_predicted_label(interest_classification_output)
            label1_predictions = logit_to_predicted_label(label1_classification_output)
            label2_predictions = logit_to_predicted_label(label2_classification_output)

            all_test_targets += test_targets.tolist()
            all_test_predictions += test_predictions.tolist()

            batch_results = pd.DataFrame({'sent_ids': sent_ids.squeeze().numpy(),
                                          'target_label': test_targets.squeeze().numpy(),
                                          'predicted_label': test_predictions.squeeze().numpy(),
                                          'is_correct': (test_targets == test_predictions).squeeze().numpy(),
                                          'label1_predictions': label1_predictions.squeeze(),
                                          'label2_predictions': label2_predictions.squeeze()})
            batch_results = batch_results.set_index('sent_ids', drop=True)
            batch_results.index.name = None  # remove index column name

            batch_data = df.loc[torch.squeeze(sent_ids)]
            batch_data = batch_data[['label1', 'label2', 'ent1', 'ent2', 'masked_sent']]

            batch_df = pd.concat([batch_results, batch_data], axis=1)

            for column in ['label1', 'label2', 'label1_predictions', 'label2_predictions']:
                batch_df[column] = le.inverse_transform(batch_df[column])

            out_df = pd.concat([out_df, batch_df], ignore_index=False)

    avg_test_loss, avg_test_labels_acc, avg_test_interest_acc = calc_measurements(model, test_dataloader, interest_criterion, labels_criterion)

    test_classification_report = classification_report(all_test_targets, all_test_predictions, zero_division=1)

    out_df.to_csv(f'{output_dir}\\test_predictions_results.csv', index=True)

    with open(f'{output_dir}\\test_report.txt', 'w') as file:
        file.write(f'Test average loss: {avg_test_loss}.\n')
        file.write(f'Test labels prediction accuracy: {avg_test_labels_acc}.\n')
        file.write(f'Test classification report:\n')
        file.write(test_classification_report)

    accuracy = accuracy_score(all_test_targets, all_test_predictions)
    return accuracy

def run_experiment(df, score,
                   labels_pred_hidden_layers_config, interest_pred_hidden_layers_config,
                   learning_rate, batch_size, num_epochs,
                   le, output_dir):
    total_start_time = time.time()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Preparing the data
    data_columns = list(df.columns[0:BERT_OUTPUT_SHAPE]) + ['label1', 'label2']
    X_train, X_tmp, y_train, y_tmp = train_test_split(df.loc[:, data_columns], df[score], random_state=42, test_size=0.4)
    y_train, y_tmp = score_to_label(y_train, y_tmp)

    X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, random_state=42, test_size=0.5)
    train_dataloader = create_data_loader(X_train, y_train, batch_size, shuffle=True)
    validation_dataloader = create_data_loader(X_val, y_val, batch_size, shuffle=False)
    test_dataloader = create_data_loader(X_test, y_test, batch_size, shuffle=False)

    # Preparing the model
    model = BERT_Classifier(input_dim=BERT_OUTPUT_SHAPE,
                            labels_pred_hidden_layers_config=labels_pred_hidden_layers_config,
                            interest_pred_hidden_layers_config=interest_pred_hidden_layers_config,
                            num_labels=len(le.classes_))
    model.to(device)

    interest_criterion = nn.CrossEntropyLoss()

    # Preparing the loss: due to data imbalance, we will use weighted loss function instead of the out-of-the-box BERT's.
    # reference: https://discuss.huggingface.co/t/class-weights-for-bertforsequenceclassification/1674/6
    train_labels = X_train[['label1', 'label2']].stack().droplevel(1)
    weight = calc_weight(train_labels)
    labels_criterion = nn.CrossEntropyLoss(weight=weight, reduction='mean')

    # Preparing the optimizer
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    # train model
    print('Start training...')
    train_results = train_model(model, optimizer, num_epochs, train_dataloader, validation_dataloader, interest_criterion, labels_criterion, output_dir)

    # test model
    print('Start testing...')
    test_acc = test_model(model, test_dataloader, df, interest_criterion, labels_criterion, le, output_dir)

    total_time = time.time() - total_start_time
    print(f'Done. total time: {total_time} seconds.\n')

    with open(f'{output_dir}\\total_time.txt', 'a') as file:
        file.write(f'Total time: {total_time} seconds.')

    experiment_results = {'max_train_acc': train_results['avg_train_acc'].max(),
                          'max_train_acc_epoch': train_results['avg_train_acc'].idxmax(),
                          'max_train_labels_acc': train_results['avg_train_labels_acc'].max(),
                          'max_train_labels_acc_epoch': train_results['avg_train_labels_acc'].idxmax(),
                          'max_val_acc': train_results['avg_val_acc'].max(),
                          'max_val_acc_epoch': train_results['avg_val_acc'].idxmax(),
                          'max_val_labels_acc': train_results['avg_val_labels_acc'].max(),
                          'max_val_labels_acc_epoch': train_results['avg_val_labels_acc'].idxmax(),
                          'test_acc': test_acc}
    return experiment_results

if __name__ == '__main__':
    if len(sys.argv) < 4:
        raise ValueError(f'Not enough arguments. Arguments given: {sys.argv[1:]}')

    input_file = sys.argv[1]  # e.g. '../data/dummy/dummy_data.csv' or '../data/data.csv'
    embeddings_file = sys.argv[2]  # e.g. '../data/dummy/embeddings_dummy.out' or '../data/embeddings.out'
    result_dir = sys.argv[3]  # e.g. 'results'

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    experiment_log_file_path = f'{result_dir}\\experiments_logs.csv'
    exp_index = init_experiment_log_file(experiment_log_file_path, CONFIG_HEADER, RESULTS_HEADER)

    num_epochs = 40

    networks_config_experiment_count = 3

    input_df = create_df(input_file, embeddings_file)

    # encode the labels
    labels = get_all_possible_labels(input_df)
    le = LabelEncoder() # encode labels from string ('GPE', 'ORG', ...) to ordinal numbers (0, 1, ...)
    le.fit(labels)
    input_df['label1'] = le.transform(input_df['label1'])
    input_df['label2'] = le.transform(input_df['label2'])

    for score in ['mi_score', 'pmi_score']:
        for learning_rate in [0.01, 0.05, 0.001, 0.005]:
            for batch_size in [256, 128, 64]:
                for i in range(networks_config_experiment_count):
                    labels_pred_hidden_layers_config = draw_hidden_layers_config()
                    interest_pred_hidden_layers_config = draw_hidden_layers_config()
                    output_dir = f'{result_dir}\\{exp_index}'
                    experiment_settings = {
                                            'exp_index': exp_index,
                                            'score': score,
                                            'labels_pred_hidden_layers_config': labels_pred_hidden_layers_config,
                                            'interest_pred_hidden_layers_config': interest_pred_hidden_layers_config,
                                            'learning_rate': learning_rate,
                                            'batch_size': batch_size,
                                            'num_epochs': num_epochs
                                           }
                    experiment_settings_str = ','.join([f'{key}={value}' for key, value in experiment_settings.items()])
                    print('running: ' + experiment_settings_str)
                    experiment_results = run_experiment(input_df, score,
                                   labels_pred_hidden_layers_config, interest_pred_hidden_layers_config,
                                   learning_rate, batch_size, num_epochs,
                                   le,
                                   output_dir)
                    log_experiment(experiment_log_file_path, CONFIG_HEADER, experiment_settings, RESULTS_HEADER,
                                   experiment_results)
                    exp_index += 1