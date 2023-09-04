import pandas as pd
import sys
from transformers import BertTokenizerFast, AutoModel
import torch
import time

OUTPUT_FILE = 'embeddings.out'

if len(sys.argv) == 1:
    raise ValueError("Path to dataset missing")
else:
    input_file = sys.argv[1]

BERT_MODEL = 'bert-base-cased'

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    start_time = time.time()

    file = open(OUTPUT_FILE, "w")

    df = pd.read_csv(input_file)
    df = df[['sent_id', 'masked_sent']]
    n = len(df)

    bert_model = AutoModel.from_pretrained(BERT_MODEL).to(device)
    tokenizer = BertTokenizerFast.from_pretrained(BERT_MODEL)

    print('Start encoding...')

    for i, (index, row) in enumerate(df.iterrows(), start=1):
        sent_id = row['sent_id']
        masked_sent = row['masked_sent']

        encoded_inputs = tokenizer(masked_sent, padding=True, truncation=True, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = bert_model(**encoded_inputs)

        embedding = outputs.last_hidden_state[:, 0, :]

        emb_lst = embedding.tolist()[0]
        emb_str = ' '.join([str(num) for num in emb_lst])

        file.write(f'{str(sent_id)} {emb_str}\n')

        if i % 100 == 0:
            print(f'total entries processed: {i}/{n} = {i / n :.2%}')

    file.close()

    print(f'Total time: {time.time() - start_time} seconds')