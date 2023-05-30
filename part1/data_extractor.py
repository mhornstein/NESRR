import spacy
from collections import Counter
import os
import csv
import time

'''
We load spacy and disable irrelevant component for NER extraction
reference: https://stackoverflow.com/questions/66613770/how-to-optimize-spacy-pipe-for-ner-only-using-an-existing-model-no-training
'''
nlp = spacy.load("en_core_web_lg", disable=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer"])

# INPUT_DIR = '../data/dummy'
INPUT_DIR = '../data/wikitext-103-raw'
OUTPUT_FILE = 'entities.csv'

K = 1000

'''
The supported entities types:

GPE: Countries, cities, states
WORK_OF_ART: Titles of books, songs, etc.
EVENT: Named hurricanes, battles, wars, sports events, etc.
FAC: Buildings, airports, highways, bridges, etc.
ORG: Companies, agencies, institutions, etc.
NORP: Nationalities or religious or political groups
PERSON: People, including fictional
PRODUCT: Objects, vehicles, foods, etc. (not services)
LOC: Non-GPE locations, mountain ranges, bodies of water
LAW: Named documents made into laws.
'''
ENTITIES_TYPES = {'FAC', 'EVENT', 'PERSON', 'ORG', 'NORP', 'PRODUCT', 'GPE', 'WORK_OF_ART', 'LAW', 'LOC'}

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

def count_entities_in_texts(texts):
    entities_counter = Counter()
    # Use 4 processes for optimal running time. Reference: https://spacy.io/usage/processing-pipelines
    for i, doc in enumerate(nlp.pipe(texts, n_process=4)):
        print(f'{i}: {doc}', end='')
        entities = doc.ents
        for entity in entities:
            if entity.label_ in ENTITIES_TYPES:
                entities_counter[entity.text] += 1
    return entities_counter

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

    entities_count = count_entities_in_texts(texts)
    print(f'Total entities found: {len(entities_count)}')

    print(f"Entities extraction and count time: {time.time() - start_time} seconds")

    '''
    Step 3: remove entities that appear less than K
    '''
    start_time = time.time()

    filtered_entities_count = Counter({key: value for key, value in entities_count.items() if value >= K})
    print(f'entities >= {K} found: {len(filtered_entities_count)}')

    with open(OUTPUT_FILE, 'w', newline='', encoding='utf8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['noun', 'count'])
        for key, value in filtered_entities_count.items():
            writer.writerow([key, value])

    print(f"Calculating top K time: {time.time() - start_time} seconds")