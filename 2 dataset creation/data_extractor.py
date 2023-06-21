import spacy
from collections import Counter
import os
import time
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
plt.switch_backend('agg') # Reference: https://stackoverflow.com/questions/52839758/matplotlib-and-runtimeerror-main-thread-is-not-in-main-loop
from itertools import count
import random
import math
import csv
from itertools import combinations
import sys

if len(sys.argv) == 1:
    raise ValueError("Path to WikiText-103 dataset missing")
else:
    input_file = sys.argv[1]

K = 1000 # Only entities with over K occurrences will be kept along with their frequency counts.
N_PROCESS = 4 # number of processes for the Spacy processing
TEXT_BATCH_SIZE = 100 # A message will be presented in the console each TEXT_BATCH_SIZE processed sentences to illustrate the progress of the processing
N = 100000 # number of sentences to sample

DATASET_FILE = 'data.csv'  # The name of the sampled dataset file

MASK_LABEL = '[MASK]'
PROBABILITY_EPSILON = 0.001
MI_EPSILON = 1e-9

NULL_ENTITY = ('null entity', 'null label') # for entities that are alone in a sentence, we'll say their paired with the "null entity"

class SentenceData:
    def __init__(self, id, txt, entities):
        self.id = id
        self.txt = txt
        self.entities = entities

    def filter_entities(self, entities_to_keep):
        self.entities = {ent for ent in self.entities if ent[0] in entities_to_keep}

'''
We load spacy and disable irrelevant component for NER extraction
reference: https://stackoverflow.com/questions/66613770/how-to-optimize-spacy-pipe-for-ner-only-using-an-existing-model-no-training
'''
nlp = spacy.load("en_core_web_lg", disable=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer"])
nlp.add_pipe('sentencizer')
'''
The supported entities types:

GPE: Countries, cities, states
WORK_OF_ART: Titles of books, songs, etc.
EVENT: Named hurricanes, battles, wars, sports events, etc.
FAC: Buildings, airports, highways, bridges, etc.
ORG: Companies, agencies, institutions, etc.
PERSON: People, including fictional
PRODUCT: Objects, vehicles, foods, etc. (not services)
LOC: Non-GPE locations, mountain ranges, bodies of water
LAW: Named documents made into laws.

note: I chose not to add NORP: Nationalities or religious or political groups
'''
ENTITIES_TYPES = {'FAC', 'EVENT', 'PERSON', 'ORG', 'PRODUCT', 'GPE', 'WORK_OF_ART', 'LAW', 'LOC', 'LANGUAGE'}
COLOR_MAP = {entity: color for entity, color in zip(ENTITIES_TYPES, sns.color_palette('Set3', len(ENTITIES_TYPES)))}

def load_texts(file_path):
    f = open(file_path, "r", encoding="utf8")
    lines = f.read().splitlines()
    f.close()
    return lines

def text_to_sentence_data(texts):
    sentences_data = {}
    id_counter = count(start=1)

    for i, doc in enumerate(nlp.pipe(texts, n_process=N_PROCESS), start=1): # Use N_PROCESS for optimal running time. Reference: https://spacy.io/usage/processing-pipelines
        if i % TEXT_BATCH_SIZE == 0:
            print(f'total texts processed: {i}/{len(texts)} = {i/len(texts) :.2%}')

        for sent in doc.sents:
            entities = sent.ents
            filtered_entities = {(entity.text, entity.label_) for entity in entities if entity.label_ in ENTITIES_TYPES}
            if len(filtered_entities) == 0:
                continue
            else:
                sent_id = next(id_counter)
                sentences_data[sent_id] = SentenceData(id=sent_id, txt=str(sent), entities=filtered_entities)

    return sentences_data

def extract_probs(sentences_data):
    '''
    This function returns 2 probabilities calculation:
    * entities_prob - entities_prob[x] = the probability of entity x to be part of a pair
    * pairs_prob - pairs_prob[(x,y)] = the probability of lexicographic-sorted pair (x,y) to appear as a pair

    The function assumes each sentence contains at least one entity

    Note that while all entries in pairs_prob are i.i.d, the case is not the same for entities_prob is not.
    A basic example to illustrate why:
    if we have just one pair (x,y), pairs_prob[(x,y)] = 1, but entities_prob[x] = entities_prob[y] = 1.
    So, x and y are not independent - they are dependent on each other.
    '''
    entities_prob = Counter()
    pairs_prob = Counter()

    for sent_data in sentences_data.values():
        entities = sent_data.entities

        if len(entities) == 1:
            entity = next(iter(entities))
            sorted_entities = [entity, NULL_ENTITY]
        else:
            sorted_entities = sorted(entities, key=lambda x: x[0])  # Keep lexicographic order

        all_pairs = list(combinations(sorted_entities, 2))
        for ent1, ent2 in all_pairs:
            pairs_prob[(ent1[0], ent2[0])] += 1

        n = len(sorted_entities)
        for ent in sorted_entities:
            entities_prob[ent[0]] += n - 1 # this is the number of possible pairs ent is part of

    # Now we convert counts => to probabilities

    n = sum(entities_prob.values())
    for key in entities_prob:
        entities_prob[key] = (entities_prob[key] * 2) / n # ent has 2 options: to be either the first or the second in the pair

    n = sum(pairs_prob.values())
    for key in pairs_prob:
        pairs_prob[key] /= n

    return entities_prob, pairs_prob

def extract_sentences_stats(sentences_data):
    entities_count = Counter()
    labels_count = Counter()
    pairs_count = Counter()
    pairs_labels_count = Counter()

    for sentence_data in sentences_data.values():
        entities = sorted(sentence_data.entities, key=lambda x: x[0]) # Keep lexicographic order
        pairs = list(combinations(entities, 2))
        for ent1, ent2 in pairs:
            pairs_count[(ent1[0], ent2[0])] += 1
            pairs_labels_count[(ent1[1], ent2[1])] += 1
            pairs_labels_count[(ent2[1], ent1[1])] += 1

        for ent_txt, ent_label in entities:
            entities_count[ent_txt] += 1
            labels_count[ent_label] += 1

    return entities_count, labels_count, pairs_count, pairs_labels_count

def plot_heatmap(df, output_file, title):
    plt.figure(figsize=(8, 6))
    columns = df.columns
    data = df.pivot(index=columns[0], columns=columns[1], values=columns[2])
    data = data.fillna(0) # Replace NaN values with 0
    sns.heatmap(data, annot=True, fmt='g')
    plt.title(title)
    plt.subplots_adjust(left=0.15, bottom=0.19) # Adjust the margins
    plt.savefig(output_file)
    plt.clf()
    plt.close()

def plot_barchart(df, output_file, title):
    colors = [COLOR_MAP.get(value, 'gray') for value in df.iloc[:, 0]]
    ax = sns.barplot(x=df.columns[0], y=df.columns[1], data=df, palette=colors)
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2, p.get_height()), ha='center', va='bottom')
    ax.tick_params(axis='x', labelsize=6)
    plt.title(title)
    plt.savefig(output_file)
    plt.clf()
    plt.close()

def plot_kde(s, output_file, title):
    sns.kdeplot(data=s)
    plt.title(title)
    plt.savefig(output_file)
    plt.clf()
    plt.close()

def pair_count_to_df(counter, columns):
    df = pd.DataFrame(counter.items(), columns = ['tmp'] + [columns[-1]])
    df[columns[0:-1]] = pd.DataFrame(df['tmp'].tolist())
    df = df.drop('tmp', axis=1)
    df = df[columns] # reorder columns
    df.sort_values(by=columns[-1], inplace=True, ascending=False)  # sort in reverse order to ease browsing
    return df

def count_to_df(counter, columns):
    df = pd.DataFrame.from_dict(counter, orient='index', columns=[columns[-1]]).reset_index()
    df.columns = columns
    df.sort_values(by=columns[-1], inplace=True, ascending=False) # sort in reverse order to ease browsing
    return df

def sample_sentences(sentences_data, n):
    ids = list(sentences_data.keys())
    random.shuffle(ids)
    sampled_ids = ids[:n]
    return {i: sentences_data[i] for i in sampled_ids}

def sample_entities_in_sentences(sentences_data):
    for sentence_data in sentences_data.values():
        entities = sentence_data.entities
        entities = random.sample(list(entities), 2)
        sentence_data.entities = entities

def calc_mi_score(ent1, ent2, entities_prob, pairs_prob):
    ent1, ent2 = (ent1, ent2) if ent1 < ent2 else (ent2, ent1) # Keep lexicographic order

    p_ent1_1 = entities_prob[ent1] # probability for ent1 to appear in a pair (success = 1)
    p_ent1_0 = 1-p_ent1_1 # probability for ent1 not to appear in a pair (success = 0)

    p_ent2_1 = entities_prob[ent2] # same as above - but for ent2
    p_ent2_0 = 1-p_ent2_1

    p_ent1_1_p_ent2_1 = pairs_prob[(ent1, ent2)] # probability for both entities to appear in a pair (success = 1).
    # Below, 0 follows bernouli "failure", i.e. the entity didn't appear in the pair
    p_ent1_1_p_ent2_0 = p_ent1_1 - p_ent1_1_p_ent2_1 # P(A ∩ B') = P(A) – P(A ∩ B)
    p_ent1_0_p_ent2_1 = p_ent2_1 - p_ent1_1_p_ent2_1
    p_ent1_0_p_ent2_0 = 1 - (p_ent1_1 + p_ent2_1 - p_ent1_1_p_ent2_1) # P(A' ∩ B') = 1 – P(A ∪ B) = 1 - [P(A) + P(B) - P(A∩B)]

    # Note: we add MI_EPSILON to handle 0 in the log
    return p_ent1_0_p_ent2_0 * math.log2((p_ent1_0_p_ent2_0 + MI_EPSILON) / (p_ent1_0 * p_ent2_0)) + \
           p_ent1_0_p_ent2_1 * math.log2((p_ent1_0_p_ent2_1 + MI_EPSILON) / (p_ent1_0 * p_ent2_1)) + \
           p_ent1_1_p_ent2_0 * math.log2((p_ent1_1_p_ent2_0 + MI_EPSILON) / (p_ent1_1 * p_ent2_0)) + \
           p_ent1_1_p_ent2_1 * math.log2((p_ent1_1_p_ent2_1 + MI_EPSILON) / (p_ent1_1 * p_ent2_1))

def calc_pmi_score(ent1, ent2, entities_count, pairs_count, n_entities, n_pairs):
    ent1, ent2 = (ent1, ent2) if ent1 < ent2 else (ent2, ent1)  # Keep lexicographic order

    p_ent1 = entities_count[ent1] / n_entities
    p_ent2 = entities_count[ent2] / n_entities
    p_ent1_ent2 = pairs_count[(ent1, ent2)] / n_pairs

    return math.log2(p_ent1_ent2) / (p_ent1 * p_ent2)

def create_dataset(sentences_data, entities_prob, pairs_prob, entities_count, pairs_count, output_file):
    csvfile = open(output_file, 'w', newline='', encoding='utf8')
    writer = csv.writer(csvfile)
    writer.writerow(['sent_id', 'sent', 'masked_sent', 'ent1', 'label1', 'ent2', 'label2', 'mi_score', 'pmi_score'])
    n_entities = sum(entities_count.values())
    n_pairs = sum(pairs_count.values())

    for sent_id in sorted(list(sentences_data.keys())):
        sent_data = sentences_data[sent_id]
        sent = sent_data.txt
        ent1, label1 = sent_data.entities[0]
        ent2, label2 = sent_data.entities[1]
        mi_score = calc_mi_score(ent1, ent2, entities_prob, pairs_prob) # mi score required the lexicographic order as present in the counts
        pmi_score = calc_pmi_score(ent1, ent2, entities_count, pairs_count, n_entities, n_pairs)

        # masking
        if sent.find(ent1) > sent.find(ent2): # switch entities order to fit the sentence order
            ent1, label1 = sent_data.entities[1]
            ent2, label2 = sent_data.entities[0]

        s1 = sent.find(ent1)
        e1 = s1 + len(ent1)

        s2 = sent.find(ent2)
        e2 = s2 + len(ent2)

        masked_sent = sent[0:s1] + MASK_LABEL + sent[e1:s2] + MASK_LABEL + sent[e2:]

        entry = [sent_id, sent, masked_sent, ent1, label1, ent2, label2, mi_score, pmi_score]
        writer.writerow(entry)

    csvfile.close()

def plot_stats(entities_count, labels_count, pairs_count, pairs_labels_count, output_dir):
    if not os.path.exists(f'./{output_dir}'):
        os.makedirs(f'{output_dir}')

    # entities
    entities_df = count_to_df(entities_count, ['entity', 'count'])
    entities_df.to_csv(f'{output_dir}\\entities_count.csv', index=False)
    plot_kde(entities_df['count'], f'{output_dir}\\entities_count.png', "Entities count")

    # entities pairs
    pairs_df = pair_count_to_df(pairs_count, ['ent1', 'ent2', 'count'])
    pairs_df.to_csv(f'{output_dir}\\pairs_count.csv', index=False)
    plot_kde(pairs_df['count'], f'{output_dir}\\pairs_count.png', "Pairs count")

    # labels
    labels_df = count_to_df(labels_count, ['label', 'count'])
    labels_df.to_csv(f'{output_dir}\\labels_count.csv', index=False)
    plot_barchart(labels_df, f'{output_dir}\\labels_count.png', "Labels count")

    # labels pairs
    pairs_labels_df = pair_count_to_df(pairs_labels_count, ['label1', 'label2', 'count'])
    pairs_labels_df.to_csv(f'{output_dir}\\pairs_labels_count.csv', index=False)
    plot_heatmap(pairs_labels_df, f'{output_dir}\\pairs_labels_count.png', 'labels pairs co-occurances')

def validate_probability(entities_prob, pairs_prob):
    '''
    Picks a Simple Random Sample of 100 entities (and if entities count < 100 - takes as much as possible).
    Returns iff for every picked entity x, p(x) = sum(p(x,y)) for every entity y that appear with x in a pair.
    Otherwise - raises an exception
    '''
    n = min(100, len(entities_prob))
    sampled_entities = random.sample(list(entities_prob.keys()), n)
    for ent in sampled_entities:
        p_ent = entities_prob[ent]
        marginal_probability = 0
        for ent1, ent2 in pairs_prob.keys():
            if ent1 == ent or ent2 == ent:
                marginal_probability += pairs_prob[(ent1, ent2)]

        diff = abs(marginal_probability - p_ent)
        if diff > PROBABILITY_EPSILON: # Use epsilon for Python's inaccuracy in the calculation
            raise ValueError(f'Incorrect probability for entity: {ent}. Marginal: {marginal_probability}, P_ent: {p_ent}. Diff: {diff}')

if __name__ == '__main__':
    start_time = time.time()

    texts = load_texts(input_file) # We load all texts together so we will process all sentences using Spacy all at once (this will speed things up). Reference: https://github.com/explosion/spaCy/discussions/8402
    print(f'Total lines extracted (containing one sentence or more): {len(texts)}')

    # Step 1: collect measurements according to the entire data

    sentences_data = text_to_sentence_data(texts)

    entities_count, labels_count, pairs_count, pairs_labels_count = extract_sentences_stats(sentences_data)
    plot_stats(entities_count, labels_count, pairs_count, pairs_labels_count, output_dir='original_sentences_stats')

    entities_prob, pairs_prob = extract_probs(sentences_data)
    validate_probability(entities_prob, pairs_prob)

    # Step 2: Create the sample: sample a subset of the relevant sentences + sample pairs of entities in this sentences

    filtered_entities_set = {key for key, value in entities_count.items() if value >= K}
    for sentence_data in sentences_data.values():
        sentence_data.filter_entities(filtered_entities_set)
    sentences_data = { i: s_data for i, s_data in sentences_data.items() if len(s_data.entities) >= 2 }
    print(f'Entities >= {K} found: {len(filtered_entities_set)}')
    print(f'Number of sentences left after filtering entities < K: {len(sentences_data)}')

    sentences_data = sample_sentences(sentences_data, N)
    sample_entities_in_sentences(sentences_data)

    sample_entities_count, sample_labels_count, sample_pairs_count, sample_pairs_labels_count = extract_sentences_stats(sentences_data)
    assert sum(sample_entities_count.values()) == 2 * N
    assert sum(sample_labels_count.values()) == 2 * N
    assert sum(sample_pairs_count.values()) == N
    assert sum(sample_pairs_labels_count.values()) == 2 * N
    plot_stats(sample_entities_count, sample_labels_count, sample_pairs_count, sample_pairs_labels_count, output_dir='sampled_sentences_stats')

    # Step 3: create the dataset according to the samples and the measurements taken

    create_dataset(sentences_data, entities_prob, pairs_prob, entities_count, pairs_count, DATASET_FILE)

    print(f"Dataset creation total time: {time.time() - start_time} seconds")