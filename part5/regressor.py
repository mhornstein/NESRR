from sklearn.model_selection import train_test_split
from transformers.optimization import AdamW
import torch
from torch.utils.data import TensorDataset, DataLoader
import time
import warnings
warnings.filterwarnings("ignore", category=FutureWarning) # Disable the warning
import os
from sklearn.preprocessing import LabelEncoder
import sys
from sklearn import metrics

sys.path.append('../')
from common.regressor_util import *

CLASSIFICATION_NETWORK_HIDDEN_LAYERS_CONFIG = [64, None] # Other example: [20, None, 9, 0.2]
REGRESSION_NETWORK_HIDDEN_LAYERS_CONFIG = [] # No hidden layers. Other examples: [512, 0.1, 128, None]

BATCH_SIZE = 32
LEARNING_RATE = 1e-5
NUM_EPOCHS = 10

MI_TRANSFORMATION = None # can be either None, or: 'sqrt', 'ln', 'log10'

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

    def __init__(self, input_dim, num_labels, classification_network_hidden_layers_config, regression_network_hidden_layers_config):
        super(BERT_Regressor, self).__init__()
        self.input_dim = input_dim
        self.num_labels = num_labels

        # classification layer
        self.classification_network = create_network(input_dim=self.input_dim,
                                                     hidden_layers_config=classification_network_hidden_layers_config,
                                                     output_dim=num_labels * 2)
        self.regression_network = create_network(input_dim=num_labels * 2+ self.input_dim,
                                                 hidden_layers_config=regression_network_hidden_layers_config,
                                                 output_dim=1)

    def forward(self, embs):
        # Step 1: classification
        x = self.classification_network(embs)

        label1_classification_output = x[:, :self.num_labels]
        label2_classification_output = x[:, self.num_labels:]

        # Step 2: regression with the classification output
        combined_input = torch.cat((label1_classification_output, label2_classification_output, embs), dim=1)
        regression_output = self.regression_network(combined_input)

        # Step 3: return the results
        return label1_classification_output, label2_classification_output, regression_output

####################

def create_df(data_file, embs_file, mi_transformation):
    # Load input data file
    df = pd.read_csv(data_file)
    df['mi_score'] = transform_mi(df['mi_score'], mi_transformation)

    df = df.set_index('sent_id')

    # Load embeddings file
    embs = pd.read_csv(embs_file, sep=' ', header=None)
    embs = embs.set_index(embs.columns[0])  # set sentence id as the index of the dataframe

    # combine both for a single df
    df = pd.concat([embs, df], axis=1)

    df = df[[col for col in df.columns if col != 'mi_score'] + ['mi_score']] # move mi to be the last column

    return df

def get_all_possible_labels(df):
    label1_values = set(df['label1'].unique())
    label2_values = set(df['label2'].unique())
    labels = list(label1_values | label2_values)
    return labels

def create_data_loader(X, y, batch_size, shuffle):
    ids_tensor = torch.tensor(X.index, dtype=torch.int64).unsqueeze(dim=1).to(device)
    embs_tensor = torch.tensor(X.iloc[:, :BERT_OUTPUT_SHAPE].values, dtype=torch.float32).to(device)
    label1_tensor = torch.tensor(X['label1'].values, dtype=torch.long).to(device) # Note that target class must be of type torch.long
    label2_tensor = torch.tensor(X['label2'].values, dtype=torch.long).to(device)
    mi_tensor = torch.tensor(y.values.reshape(-1, 1), dtype=torch.float32).to(device)
    dataset = TensorDataset(ids_tensor, embs_tensor, label1_tensor, label2_tensor, mi_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader

def get_classification_report(header, true_labels, predicted_labels):
    '''
    Generate a report as a string containing scores for the classification.
    Notes:
        The averaging over the classes is set to 'weighted', which means the average is calculated based on
        the number of true instances for each label (= "support")
        This may result in an F-score that is not between precision and recall.
    :param header: title for the classification report
    :param true_labels: True class labels
    :param predicted_labels: Predicted class labels
    :return: str: report as a string
    '''
    accuracy = metrics.accuracy_score(true_labels, predicted_labels)
    precision = metrics.precision_score(true_labels, predicted_labels, average='weighted', zero_division=0)
    recall = metrics.recall_score(true_labels, predicted_labels, average='weighted', zero_division=0)
    f1_score = metrics.f1_score(true_labels, predicted_labels, average='weighted', zero_division=0)
    classification_report = metrics.classification_report(true_labels, predicted_labels, zero_division=0)

    return f'classification scores for {header}\n' \
           f'----------------------------------\n' \
           f'Accuracy: {accuracy}\n' \
           f'Precision: {precision}\n' \
           f'Recall: {recall}\n' \
           f'F1 Score: {f1_score}\n' \
           f'\nClassification Report:\n{classification_report}'

def get_encoding_report(le):
    '''
    This function generates a report of labels and their corresponding encodings using a fitted LabelEncoder object.
    :param le: A fitted LabelEncoder object. It should be previously fitted on the labels.
    :return: A string containing the report of labels and their encodings. \
            The string is formatted with the label and its corresponding encoding in a tabular format.
    '''
    s =  'encodings:\n'
    s += '----------\n'
    s += 'Label\t\tEncoding\n'
    for label, encoding in zip(le.classes_, le.transform(le.classes_)):
        s += f'{label: <12}\t{encoding}\n'
    return s

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = create_df(data_file=input_file, embs_file=embeddings_file, mi_transformation=MI_TRANSFORMATION)

    labels = get_all_possible_labels(df)

    le = LabelEncoder() # encode labels from string ('GPE', 'ORG', ...) to ordinal numbers (0, 1, ...)
    le.fit(labels)
    df['label1'] = le.transform(df['label1'])
    df['label2'] = le.transform(df['label2'])

    X_train, X_tmp, y_train, y_tmp = train_test_split(df.iloc[:, :-1], df['mi_score'], random_state=42, test_size=0.3)
    X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, random_state=42, test_size=0.5)
    train_dataloader = create_data_loader(X_train, y_train, BATCH_SIZE, shuffle=True)
    validation_dataloader = create_data_loader(X_val, y_val, BATCH_SIZE, shuffle=True)
    test_dataloader = create_data_loader(X_test, y_test, BATCH_SIZE, shuffle=False)

    model = BERT_Regressor(input_dim=BERT_OUTPUT_SHAPE,
                           num_labels=len(labels),
                           classification_network_hidden_layers_config=CLASSIFICATION_NETWORK_HIDDEN_LAYERS_CONFIG,
                           regression_network_hidden_layers_config=REGRESSION_NETWORK_HIDDEN_LAYERS_CONFIG)
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
        for ids, embeddings, labels1, labels2, mi_score in train_dataloader:
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
            for val_ids, val_embeddings, val_labels1, val_labels2, val_mi_score in validation_dataloader:
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
        
    results_to_files(results_dict=results, output_dir=OUTPUT_DIR)

    print('Start testing...')
    model.eval()

    out_df = pd.DataFrame(columns=['target_mi', 'predicted_mi', 'abs_mi_err',
                                   'ent1', 'label1', 'label1_pred',
                                   'ent2', 'label2', 'label2_pred',
                                   'masked_sent'])

    out_df = out_df.astype({'target_mi': float, 'predicted_mi': float, 'abs_mi_err': float,
                            'ent1': str, 'label1': int, 'label1_pred': int,
                            'ent2': str, 'label2': int, 'label2_pred': int,
                            'masked_sent': str}) # Assign the desired data types to the columns
    test_total_loss = 0
    with torch.no_grad():
        for test_ids, test_embeddings, test_labels1, test_labels2, test_mi_score in test_dataloader:
            test_label1_classification_output, test_label2_classification_output, test_regression_output = model(test_embeddings)

            # Calculate loss: here we do not care about the classification loss but only care about the regression loss
            test_regression_loss = regression_criterion(test_regression_output, test_mi_score)
            test_total_loss += test_regression_loss.item()

            # log results
            absolute_errors = torch.abs(test_regression_output - test_mi_score)
            label1_preds = torch.max(test_label1_classification_output, 1)[1]
            label2_preds = torch.max(test_label2_classification_output, 1)[1]
            batch_data = df.loc[torch.squeeze(test_ids)]
            batch_results = pd.DataFrame({'sent_ids': test_ids.squeeze().numpy(),
                                          'target_mi': test_mi_score.squeeze().numpy(),
                                          'predicted_mi': test_regression_output.squeeze().numpy(),
                                          'abs_mi_err': absolute_errors.squeeze().numpy(),
                                          'ent1': batch_data['ent1'],
                                          'label1': batch_data['label1'],
                                          'label1_pred': label1_preds.numpy(),
                                          'ent2': batch_data['ent2'],
                                          'label2': batch_data['label2'],
                                          'label2_pred': label2_preds.numpy(),
                                          'masked_sent': batch_data['masked_sent'] })
            batch_results = batch_results.set_index('sent_ids', drop=True)
            batch_results.index.name = None # remove index column name

            out_df = out_df.append(batch_results, ignore_index=False)

    avg_test_loss = test_total_loss / len(test_dataloader)

    with open(f'{OUTPUT_DIR}/test_report.txt', 'w') as file:
        file.write(f'Average test regression loss: {avg_test_loss}\n')
        file.write('\n')

        encoding_report = get_encoding_report(le)
        file.write(encoding_report)
        file.write('\n')

        label1_report = get_classification_report('label1', out_df['label1'], out_df['label1_pred'])
        file.write(label1_report)
        file.write('\n')

        label2_report = get_classification_report('label2', out_df['label2'], out_df['label2_pred'])
        file.write(label2_report)

    out_df['label1'] = le.inverse_transform(out_df['label1']) # change labels in out_df back from numbers => to string labels
    out_df['label1_pred'] = le.inverse_transform(out_df['label1_pred'])
    out_df['label2'] = le.inverse_transform(out_df['label2'])
    out_df['label2_pred'] = le.inverse_transform(out_df['label2_pred'])
    out_df.to_csv(f'{OUTPUT_DIR}/test_predictions_results.csv', index=True)