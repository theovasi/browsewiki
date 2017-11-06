""" Creates a dictionary that matches words to their roots. """
import time
import math
import argparse
import joblib

from greek_stemmer import GreekStemmer
from toolset.corpus import Corpus
from toolset.mogreltk import tokenize, normalize

def main(data_file_path, stopwords_file_path):
    crp = Corpus('{}/formatted'.format(data_file_path),
                 filepath_dict_path='{}/filepath_dict.txt'.format(data_file_path))
    doc_gen = crp.document_generator
    stemmer = GreekStemmer()
    lemma_dict = dict()
    n_docs = 0

    start_time = time.time()
    for doc in doc_gen():
        words_tokenized = tokenize(doc, stopwords_file_path)
        words_stemmed = \
                [stemmer.stem(normalize(token).upper()).lower() for token in words_tokenized]
        for index, stem in enumerate(words_stemmed):
            if not stem in lemma_dict:
                lemma_dict[stem] = [words_tokenized[index]]
            elif not words_tokenized[index] in lemma_dict[stem]:
                lemma_dict[stem].append(words_tokenized[index])
        n_docs += 1
        if n_docs%2000 == 0:
            print('Processed {} documents... - {} docs/s'.format(n_docs,
                  math.floor(n_docs / (time.time() - start_time))))

    print('Created lemma dictionary with {} items.'.format(len(lemma_dict)))
    joblib.dump(lemma_dict, '{}/lemma_dict.pd'.format(data_file_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_file_path', type=str,
                        help='The path to the data directory.')
    parser.add_argument('-s', '--stop', type=str,
                        help='The path to the stopwords file.')
    arguments = parser.parse_args()
    main(arguments.data_file_path, arguments.stop)
