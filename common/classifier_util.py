import numpy as np

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
