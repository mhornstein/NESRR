import spacy
from collections import Counter
nlp = spacy.load("en_core_web_lg")
import os
import json
import time

# INPUT_DIR = '../data/dummy'
INPUT_DIR = '../data/wikitext-103-raw'
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

Note: not supported yet might be further looked into
LAW: Named documents made into laws.
LOC: Non-GPE locations, mountain ranges, bodies of water
'''
ENTITIES_TYPES = {'FAC', 'EVENT', 'PERSON', 'ORG', 'NORP', 'PRODUCT', 'GPE', 'WORK_OF_ART', 'LAW', 'LOC'}

def get_entities(dir, k):
    entities_counter = Counter()
    for file in os.listdir(dir):
        print(f'Analyzing file: {file}')
        file_path = os.path.join(dir, file)
        f = open(file_path, "r", encoding="utf8")
        for line in f:
            if line == ' \n' or line.startswith(' = '):
                continue
            else:
                doc = nlp(line)
                entities = [ent for ent in doc.ents if ent.label_ in ENTITIES_TYPES]
                entities_counter.update(entities)
        f.close()
    return {key for key, count in entities_counter.items() if count >= k}

start_time = time.time()

entities = get_entities(INPUT_DIR, k=1000)
print(f'10 entities samples: {entities[0:10]}')
print(f'entities found: {len(entities)}')
with open("entities.json", "w") as outfile:
    outfile.write(entities)

end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time} seconds")