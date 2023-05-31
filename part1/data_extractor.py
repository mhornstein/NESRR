import spacy
from collections import Counter
import os
import csv
import time
from itertools import combinations

DEBUG = True

if DEBUG:
    INPUT_DIR = '../data/dummy'
    ENTITY_COUNT_OUTPUT_FILE = 'dummy_entities_count.csv'
    PAIRS_COUNT_OUTPUT_FILE = 'dummy_pairs_count.csv'
    LABELS_COUNT_OUTPUT_FILE = 'dummy_labels_count.csv'
    PAIRS_LABELS_COUNT_OUTPUT_FILE = 'dummy_pairs_labels_count.csv'
    K = 3
    N_PROCESS = 1
else:
    INPUT_DIR = '../data/wikitext-103-raw'
    ENTITY_COUNT_OUTPUT_FILE = 'entities_count.csv'
    PAIRS_COUNT_OUTPUT_FILE = 'pairs_count.csv'
    LABELS_COUNT_OUTPUT_FILE = 'labels_count.csv'
    PAIRS_LABELS_COUNT_OUTPUT_FILE = 'pairs_labels_count.csv'
    K = 1000
    N_PROCESS = 4

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
    texts = []
    for file in os.listdir(dir):
        print(f'Loading file: {file}')
        file_path = os.path.join(dir, file)
        f = open(file_path, "r", encoding="utf8")
        for line in f:
            if line != ' \n' and not line.startswith(' = '):
                texts.append(line)
        f.close()
    return texts

def extract_text_data(texts):
    sentences_data = {}
    entities_counter = Counter()

    for i, doc in enumerate(nlp.pipe(texts, n_process=N_PROCESS)): # Use N_PROCESS for optimal running time. Reference: https://spacy.io/usage/processing-pipelines
        print(f'{i}: {doc}', end='')
        entities = doc.ents
        filtered_entities = {(entity.text, entity.label_) for entity in entities if entity.label_ in ENTITIES_TYPES}
        if len(filtered_entities) >= 2: # Keep only sentences with at least one candidate pairs
            for ent_text, ent_label in filtered_entities:
                entities_counter[ent_text] += 1
            sentences_data[i] = SentenceData(id=i, txt=str(doc), entities=filtered_entities)
    return sentences_data, entities_counter

def count_pairs_data(sentences_data):
    labels_count = Counter()
    pairs_labels_count = Counter()
    pairs_count = Counter()

    for sentence_data in sentences_data.values():
        entities = sorted(sentence_data.entities, key=lambda x: x[0]) # Keep lexicographic order
        pairs = list(combinations(entities, 2))
        for ent1, ent2 in pairs:
            pairs_count[(ent1[0], ent2[0])] += 1
            pairs_labels_count[(ent1[1], ent2[1])] += 1
            pairs_labels_count[(ent2[1], ent1[1])] += 1

        for ent in entities:
            labels_count[ent[1]] += 1

    return labels_count, pairs_labels_count, pairs_count

def log_count(count, output_file):
    with open(output_file, 'w', encoding='utf8') as file:
        file.write(f'entry,count\n')
        sorted_entities = count.most_common() # sort by occurances for better browsing of csv
        for key, value in sorted_entities:
            file.write(f'{key},{value}\n')

if __name__ == '__main__':
    '''
    Step 1: load all the texts. 
    Reason: We process all the texts *together* due to the fact that when you create a spacy doc, 
    it requires getting and then releasing a number of resources that by default are not re-used between calls.
    Therefore, calling doc once for all texts might speed up the processing.
    Reference: https://github.com/explosion/spaCy/discussions/8402
    '''
    start_time = time.time()

    texts = load_texts(INPUT_DIR)
    print(f'Total lines extracted: {len(texts)}')

    print(f"Loading files time: {time.time() - start_time} seconds")

    '''
    Step 2: extract and count entities
    '''
    start_time = time.time()

    sentences_data, entities_count = extract_text_data(texts)
    print(f'\nTotal relevant entities found: {len(entities_count)}') # Relevant entity: entity that is of a relevant label and has another entity in the sentence

    print(f"Entities extraction and count time: {time.time() - start_time} seconds")

    '''
    Step 3: remove entities that appear less than K + log the result
    '''
    start_time = time.time()

    filtered_entities_count = Counter({key: value for key, value in entities_count.items() if value >= K})
    print(f'entities >= {K} found: {len(filtered_entities_count)}')
    log_count(filtered_entities_count, ENTITY_COUNT_OUTPUT_FILE)

    print(f"Calculating top K time: {time.time() - start_time} seconds")

    '''
    Step 4: filter out entities with too few occurances. keep only the sentences that has at least one pair of entities
    '''
    start_time = time.time()
    filtered_entities_set = set(filtered_entities_count.keys())
    for sentence_data in sentences_data.values():
        sentence_data.filter_entities(filtered_entities_set)
    sentences_data = { i: s_data for i, s_data in sentences_data.items() if len(s_data.entities) >= 2 }
    print(f'Total relevant sentences: {len(sentences_data)}')
    print(f"Removing irrelevant entities from sentences data time: {time.time() - start_time} seconds")

    '''
    Step 5: count the pairs + labels statistics 
    '''
    labels_count, pairs_labels_count, pairs_count = count_pairs_data(sentences_data)
    log_count(labels_count, LABELS_COUNT_OUTPUT_FILE)
    log_count(pairs_labels_count, PAIRS_LABELS_COUNT_OUTPUT_FILE)
    log_count(pairs_count, PAIRS_COUNT_OUTPUT_FILE)
