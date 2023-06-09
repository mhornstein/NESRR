from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers.optimization import AdamW
from torch.utils.data import TensorDataset, DataLoader
import time
import warnings
warnings.filterwarnings("ignore", category=FutureWarning) # Disable the warning
import sys
from sklearn.metrics import classification_report

sys.path.append('../')
from common.util import *
from common.classifier_util import *

BERT_MODEL = 'bert-base-uncased' #  'distilbert-base-uncased'

CONFIG_HEADER = ['exp_index', 'score', 'score_threshold_type', 'score_threshold_value', 'learning_rate', 'batch_size', 'num_epochs']

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

def calc_measurements(model, dataloader, criterion):
    total_loss = total_good = total = 0
    with torch.no_grad():
        for sent_ids, input_ids, attention_mask, targets in dataloader:
            outputs = model(input_ids, attention_mask=attention_mask)
            total_loss += criterion(outputs.logits, targets).item()
            predictions = logit_to_predicted_label(outputs.logits)
            total_good += (predictions == targets).sum().item()
            total += len(targets)

    avg_loss = total_loss / total
    avg_acc = total_good / total

    return avg_loss, avg_acc


def train_model(model, optimizer, num_epochs, train_dataloader, validation_dataloader, criterion):
    print('Evaluating beginning state... ')
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
    for epoch in range(1, num_epochs + 1):
        print(f'Starting Epoch: {epoch}/{num_epochs}\n')
        start_time = time.time()
        model.train()
        for batch_i, (sent_ids, input_ids, attention_mask, targets) in enumerate(train_dataloader, start=1):
            print(f'Training batch: {batch_i}/{len(train_dataloader)}')
            outputs = model(input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, targets)
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

def test_model(model, test_dataloader, df, criterion, output_dir):
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

            batch_df = create_batch_result_df(data_df=df, sent_ids=test_sent_ids, targets=test_targets,
                                              predictions=test_predictions, is_correct=is_correct)

            out_df = pd.concat([out_df, batch_df], ignore_index=False)

    test_size = len(all_test_targets)
    avg_test_loss = test_total_loss / test_size

    test_classification_report = classification_report(all_test_targets, all_test_predictions, zero_division=1)

    out_df.to_csv(f'{output_dir}/test_predictions_results.csv', index=True)

    with open(f'{output_dir}/test_report.txt', 'w') as file:
        file.write(f'Test average loss: {avg_test_loss}.\n')
        file.write(f'Test classification report:\n')
        file.write(test_classification_report)

def run_experiment(input_file, score, score_threshold_type, score_threshold_value, learning_rate, batch_size, num_epochs, output_dir):
    total_start_time = time.time()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Preparing the data
    df = pd.read_csv(input_file).set_index('sent_id')
    X_train, X_tmp, y_train, y_tmp = train_test_split(df['masked_sent'], df[score], random_state=42, test_size=0.4)
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

    # train model
    print('Start training...')
    train_model(model, optimizer, num_epochs, train_dataloader, validation_dataloader, criterion)

    # test model
    print('Start testing...')
    test_model(model, test_dataloader, df, criterion, output_dir)

    total_time = time.time() - total_start_time
    print(f'Done. total time: {total_time} seconds.\n')

    with open(f'{output_dir}/total_time.txt', 'w') as file:
        file.write(f'Total time: {total_time}.')

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result_dir = 'results'

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    experiment_config_file_path = f'{result_dir}/experiments_settings.csv'
    init_experiment_config_file(experiment_config_file_path, CONFIG_HEADER)

    num_epochs = 10

    exp_index = 1

    input_file = '../data/dummy/dummy_data.csv'

    for score in ['mi_score', 'pmi_score']:
        for score_threshold_type in ['percentile', 'std_dist']:
            score_thresholds = [0.25, 0.5, 0.75] if score_threshold_type == 'percentile' else [-2, -1, 1, 2]
            for score_threshold_value in score_thresholds:
                for learning_rate in [0.01, 0.05, 0.001, 0.005]:
                    for batch_size in [64, 128, 256]:
                        output_dir = f'{result_dir}/{exp_index}'
                        experiment_settings = {
                                                'exp_index': exp_index,
                                                'score': score,
                                                'score_threshold_type': score_threshold_type,
                                                'score_threshold_value': score_threshold_value,
                                                'learning_rate': learning_rate,
                                                'batch_size': batch_size,
                                                'num_epochs': num_epochs
                                               }
                        experiment_settings_str = ','.join([f'{key}={value}' for key, value in experiment_settings.items()])
                        print('running: ' + experiment_settings_str)
                        write_experiment_config(experiment_config_file_path, experiment_settings, CONFIG_HEADER)
                        run_experiment(input_file, score, score_threshold_type, score_threshold_value, learning_rate,
                                       batch_size, num_epochs, output_dir)
                        exp_index += 1