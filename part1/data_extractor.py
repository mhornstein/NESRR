import spacy
from collections import Counter
import os
import time
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import count
import random
import math
import csv
from itertools import combinations
import sys

if len(sys.argv) == 1:
    raise ValueError("Path to WikiText-103 dataset missing")
else:
    input_dir = sys.argv[1]

DEBUG = True

if DEBUG:
    K = 3
    N_PROCESS = 1
    TEXT_BATCH_SIZE = 10
    N = 15
else:
    K = 1000 # number os sentences Spacy will process in each batch
    N_PROCESS = 4 # number of processes for the Spacy processing
    TEXT_BATCH_SIZE = 100 # A message will be presented in the console each TEXT_BATCH_SIZE processed sentences to illustrate the progress of the processing
    N = 10000 # number of sentences to sample

DATASET_FILE = 'data.csv'

MASK_LABEL = '[MASK]'

class SentenceData:
    def __init__(self, id, txt, entities):
        self.id = id
        self.txt = txt
        self.entities = entities

    def filter_entities(self, entities_to_keep):
        self.entities = {ent for ent in self.entities if ent[0] in entities_to_keep}

    def get_masked_sent(self):
        t = self.txt

        ent1_txt = self.entities[0][0]
        s1 = t.find(ent1_txt)
        e1 = s1 + len(ent1_txt)

        ent2_txt = self.entities[1][0]
        s2 = t.find(ent2_txt)
        e2 = s2 + len(ent2_txt)

        if s1 < s2:
            masked_txt = t[0:s1] + MASK_LABEL + t[e1:s2] + MASK_LABEL + t[e2:]
        else:
            masked_txt = t[0:s2] + MASK_LABEL + t[e2:s1] + MASK_LABEL + t[e1:]

        return masked_txt


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
ENTITIES_TYPES = {'FAC', 'EVENT', 'PERSON', 'ORG', 'PRODUCT', 'GPE', 'WORK_OF_ART', 'LAW', 'LOC'}

def load_texts(dir):
    '''
    Loads all the .raw files from the given dir
    :return: list of texts (text = paragraph consisting of one or more sentences according to the original .raw file)
    '''
    texts = []
    for file in os.listdir(dir):
        print(f'Loading file: {file}')
        file_path = os.path.join(dir, file)
        f = open(file_path, "r", encoding="utf8")
        for line in f:
            if line != '\n' and line != ' \n' and not line.startswith(' = '):
                texts.append(line.strip())
        f.close()
    return texts

def extract_text_data(texts):
    '''
    Given list of texts, returns a dictionary mapping sentence-id -> SentenceData instance, and entities count (for future use)
    The SentenceData are created according to the information extracted from the sentences that were processed using the Spacy library
    '''

    sentences_data = {}
    entities_counter = Counter()
    id_counter = count(start=1)

    for i, doc in enumerate(nlp.pipe(texts, n_process=N_PROCESS), start=1): # Use N_PROCESS for optimal running time. Reference: https://spacy.io/usage/processing-pipelines
        if i % TEXT_BATCH_SIZE == 0:
            print(f'total texts processed: {i}/{len(texts)} = {i/len(texts) :.2%}')

        for sent in doc.sents:
            entities = sent.ents
            filtered_entities = {(entity.text, entity.label_) for entity in entities if entity.label_ in ENTITIES_TYPES}
            if len(filtered_entities) >= 2: # Keep only sentences with at least one candidate pairs
                for ent_text, _ in filtered_entities:
                    entities_counter[ent_text] += 1
                sent_id = next(id_counter)
                sentences_data[sent_id] = SentenceData(id=sent_id, txt=str(sent), entities=filtered_entities)
    return sentences_data, entities_counter

def extract_pairs_and_labels_stats(sentences_data):
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
    ax = sns.barplot(x=df.columns[0], y=df.columns[1], data=df)
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2, p.get_height()), ha='center', va='bottom')
    ax.tick_params(axis='x', labelsize=6)
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

def sample_and_sort_entities_in_sentences(sentences_data):
    for sentence_data in sentences_data.values():
        entities = sentence_data.entities
        entities = random.sample(list(entities), 2)
        entities = sorted(entities, key=lambda x: x[0])  # Keep lexicographic order after sampling
        sentence_data.entities = entities

def calc_mi_score(ent1, ent2, entities_count, pairs_count):
    '''
    calculates the mi score according to the furmula:
    MI(ent1, ent2) = P(ent1, ent2) * log2(P(ent1, ent2) / (P(ent1) * P(ent2)))
    '''
    p_ent1_ent2 = pairs_count[(ent1, ent2)] / len(pairs_count)
    p_ent1 = entities_count[ent1] / len(entities_count)
    p_ent2 = entities_count[ent2] / len(entities_count)
    mi_score = p_ent1_ent2 * math.log2(p_ent1_ent2 / (p_ent1 * p_ent2))
    return mi_score

def write_to_csv(sentences_data, output_file):
    csvfile = open(output_file, 'w', newline='', encoding='utf8')
    writer = csv.writer(csvfile)
    writer.writerow(['sent_id', 'sent', 'masked_sent', 'ent1', 'label1', 'ent2', 'label2', 'mi_score'])

    for sent_id in sorted(list(sentences_data.keys())):
        sent_data = sentences_data[sent_id]
        sent = sent_data.txt
        masked_sent = sent_data.get_masked_sent()
        ent1, label1 = sent_data.entities[0]
        ent2, label2 = sent_data.entities[1]
        mi_score = calc_mi_score(ent1, ent2, entities_count, pairs_count)
        entry = [sent_id, sent, masked_sent, ent1, label1, ent2, label2, mi_score]
        writer.writerow(entry)


def plot_stats(entities_count, labels_count, pairs_count, pairs_labels_count, output_dir):
    if not os.path.exists(f'./{output_dir}'):
        os.makedirs(f'{output_dir}')

    # entities
    entities_df = count_to_df(entities_count, ['entity', 'count'])
    entities_df.to_csv(f'{output_dir}\\entities_count.csv', index=False)

    # entities pairs
    pairs_df = pair_count_to_df(pairs_count, ['ent1', 'ent2', 'count'])
    pairs_df.to_csv(f'{output_dir}\\pairs_count.csv', index=False)

    # labels
    labels_df = count_to_df(labels_count, ['label', 'count'])
    labels_df.to_csv(f'{output_dir}\\labels_count.csv', index=False)
    plot_barchart(labels_df, f'{output_dir}\\labels_count.png', "Labels count")

    # labels pairs
    pairs_labels_df = pair_count_to_df(pairs_labels_count, ['label1', 'label2', 'count'])
    pairs_labels_df.to_csv(f'{output_dir}\\pairs_labels_count.csv', index=False)
    plot_heatmap(pairs_labels_df, f'{output_dir}\\pairs_labels_count.png', 'labels pairs co-occurances')

if __name__ == '__main__':
    start_time = time.time()

    texts = load_texts(input_dir) # We load all texts together so we will process all sentences using Spacy all at once (this will speed things up). Reference: https://github.com/explosion/spaCy/discussions/8402
    print(f'Total lines extracted (containing one sentence or more): {len(texts)}')

    sentences_data, entities_count = extract_text_data(texts)
    print(f'Total relevant entities found: {len(entities_count)}') # Relevant entity: entity that is of a relevant label and has another entity in the sentence
    print(f'Total relevant sentences found: {len(sentences_data)}') # Relevant sentence: it has at least 2 relevant entities in it

    filtered_entities_count = Counter({key: value for key, value in entities_count.items() if value >= K})
    print(f'entities >= {K} found: {len(filtered_entities_count)}')

    filtered_entities_set = set(filtered_entities_count.keys())
    for sentence_data in sentences_data.values():
        sentence_data.filter_entities(filtered_entities_set)
    sentences_data = { i: s_data for i, s_data in sentences_data.items() if len(s_data.entities) >= 2 }
    print(f'Relevant sentences after rare entities filtering: {len(sentences_data)}')

    entities_count, labels_count, pairs_count, pairs_labels_count = extract_pairs_and_labels_stats(sentences_data)
    plot_stats(entities_count, labels_count, pairs_count, pairs_labels_count, output_dir='original_data_stats')

    sentences_data = sample_sentences(sentences_data, N)
    sample_and_sort_entities_in_sentences(sentences_data)

    entities_count, labels_count, pairs_count, pairs_labels_count = extract_pairs_and_labels_stats(sentences_data)
    assert sum(entities_count.values()) == 2 * N
    assert sum(labels_count.values()) == 2 * N
    assert sum(pairs_count.values()) == N
    assert sum(pairs_labels_count.values()) == 2 * N
    plot_stats(entities_count, labels_count, pairs_count, pairs_labels_count, output_dir='sampled_data_stats')

    write_to_csv(sentences_data, DATASET_FILE)

    print(f"Dataset creation total time: {time.time() - start_time} seconds")