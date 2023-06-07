from sacremoses import MosesDetokenizer
import time
import os
import sys

if len(sys.argv) == 1:
    raise ValueError("Path to dataset dir missing")
else:
    input_dir = sys.argv[1]

OUTPUT_FILE = 'processed_data.txt'
TEXT_BATCH_SIZE = 10000 # A message will be presented in the console each TEXT_BATCH_SIZE processed sentences to illustrate the progress of the processing

detokenizer = MosesDetokenizer()

def undo_tokenization(line):
    '''
    This function reverts the preprocessing done for the dataset, as explained in the paper:
    https://arxiv.org/pdf/1609.07843.pdf

    Note: I found more tokenization issues, such as big hyphens (' – ' => '-') or slashes ('131 / M-46' = > '131/M-46'
    But as it wasn't referred to in the paper, I assume the data was crawled this way.
    '''
    l = line.replace(' @.@ ', '.') # @.@ for numbers. e.g. 1.2 => 1 @.@ 2
    l = l.replace(' @,@ ', ',') # @,@ for numbers. e.g 1,202 => 1 @,@ 202
    l = l.replace(' @-@ ', '-') # @-@ was used the escape hyphens. removing it
    detokenized_sentence = detokenizer.detokenize(l.split())
    return detokenized_sentence

if __name__ == '__main__':
    start_time = time.time()
    output_file = open(OUTPUT_FILE, "w", encoding="utf8")
    for file in os.listdir(input_dir):
        print(f'Loading file: {file}')
        file_path = os.path.join(input_dir, file)
        f = open(file_path, "r", encoding="utf8")
        lines = f.read().splitlines()
        n = len(lines)
        f.close()

        for i, line in enumerate(lines, start=1):
            if i % TEXT_BATCH_SIZE == 0:
                print(f'total texts processed: {i}/{n} = {i / n :.2%}')
            if line != '' and line != ' ' and not line.startswith(' = '):
                l = line.strip()
                l = undo_tokenization(l)
                output_file.write(l + "\n")
    output_file.close()

    print(f"Data preprocessing total time: {time.time() - start_time} seconds")