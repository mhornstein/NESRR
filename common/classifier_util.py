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