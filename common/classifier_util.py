import numpy as np
import torch
import pandas as pd

def score_to_label(y_train, y_tmp):
    train_description = y_train.describe()
    treshold = train_description['50%']

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