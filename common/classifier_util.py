import numpy as np
import torch
import pandas as pd

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

def logit_to_predicted_label(logits):
    labels = logits.argmax(1)
    return labels

def calc_weight(labels):
    '''
    Returns a tensor representing the weights for each class.
    The weights for each class are inversely proportional to the proportion of examples in that class.
    If there are fewer examples in a class, the weight for that class is increased to give more importance to those examples during the loss computation.
    We use 2 for scaling.
    Partial reference: https://forums.fast.ai/t/about-weighted-bceloss/78570/3
    '''
    total_examples = len(labels)
    unique_classes = np.unique(labels)
    class_weights = []

    for class_label in unique_classes:
        num_examples = np.sum(labels == class_label)
        class_weight = total_examples / (2 * num_examples)
        class_weights.append(class_weight)

    return torch.tensor(class_weights, dtype=torch.float32) # dtype of float 32 is the requirement of the weight

def create_batch_result_df(data_df, sent_ids, targets, predictions, is_correct):
    batch_results = pd.DataFrame({'sent_ids': sent_ids.squeeze().numpy(),
                                  'target_label': targets.squeeze().numpy(),
                                  'predicted_label': predictions.squeeze().numpy(),
                                  'is_correct': is_correct.squeeze().numpy()})
    batch_results = batch_results.set_index('sent_ids', drop=True)
    batch_results.index.name = None  # remove index column name

    batch_data = data_df.loc[torch.squeeze(sent_ids)]
    batch_data = batch_data[['label1', 'label2', 'ent1', 'ent2', 'masked_sent']]

    batch_df = pd.concat([batch_results, batch_data], axis=1)

    return batch_df