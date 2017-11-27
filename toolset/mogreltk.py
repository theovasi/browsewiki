""" This module offers a number tools for processing modern Greek text. """
import re
import time
import math
from nltk import sent_tokenize, word_tokenize
from greek_stemmer import GreekStemmer


def normalize(text):
    """ Remove intonation from Greek text.

        Args:
            text (str): The text that will be normalized.
        Returns:
            text (str): The original text in lowercase without intonation.
    """
    vowel_dict = {
        u'ά': u'α',
        u'έ': u'ε',
        u'ή': u'η',
        u'ί': u'ι',
        u'ό': u'ο',
        u'ύ': u'υ',
        u'ώ': u'ω',
        u'ϊ': u'ι',
        u'ϋ': u'υ'
    }
    text = text.lower()
    for key, value in vowel_dict.items():
        text = text.replace(key, value)

    return text


def tokenize(text, stopwords_file_path=None):
    """ Takes a String as input and returns a list of its tokens.

        Args:
            text (str): The text to be tokenized.
        Returns:
            filtered_tokens: The original text as a list of tokens excluding stopwords,
                punctuation and numbers.
    """
    filtered_tokens = []
    if stopwords_file_path is not None:
        with open(stopwords_file_path) as file_data:
            stopwords = file_data.read().splitlines()
        stopwords = [normalize(stopword) for stopword in stopwords]

    tokens = [
        word.lower()
        for sent in sent_tokenize(text)
        for word in word_tokenize(sent)
    ]
    # Remove tokens that 1) do not contain letters 2) are stopwords 3) are punctuation
    # 4) contain less than 3 letters.
    for token in tokens:
        if not re.search('^\d*$', token)\
                and not (stopwords_file_path is not None and normalize(token) in stopwords)\
                and not re.search('[\.,!;:\"\'-«»\.\.\.%]', token)\
                and len(token) > 2:
            filtered_tokens.append(token)

    return filtered_tokens


def stem(text, stopwords_file_path=None):
    """ Takes a string as input and returns a list of its stems.

        Args:
            text (str): The text to be stemmed.
        Returns:
            stems (list): The original text as a list of stems.
    """
    stemmer = GreekStemmer()
    tokens = tokenize(text, stopwords_file_path)
    stems = [stemmer.stem(normalize(token).upper()).lower()
             for token in tokens]
    return stems


class Lemmatizer(object):
    """ Offers lemmatization functionality by matching tokens to stems(many to one). """

    def __init__(self):
        self.lemma_dict = dict()

    def fit(self, text_collection_iterable, stopwords_file_path=None):
        """ Process a text collection and create a dictionary that matches stems to tokens.

            Args:
                text_collection_iterable (obj): A list, generator or other iterable object that
                    returns documents in string format.
                stopwords_file_path (str): The path to a file containing stopwords. Stopwords
                    are ignored during the processing of the collection.

        """
        print('Fitting lemmatizer...')
        stemmer = GreekStemmer()
        n_docs = 0

        start_time = time.time()
        for doc in text_collection_iterable:
            words_tokenized = tokenize(doc, stopwords_file_path)
            words_stemmed = \
                [stemmer.stem(normalize(token).upper()).lower()
                 for token in words_tokenized]
            for index, word in enumerate(words_stemmed):
                self.add(word, words_tokenized[index])

            # Print some info.
            n_docs += 1
            if n_docs % 2000 == 0:
                print('Processed {} documents... - {} docs/s'.format(n_docs,
                                                                     math.floor(n_docs / (time.time() - start_time))))

    def add(self, key, value):
        """ Add a matching of a token to a stem."""
        if not key in self.lemma_dict:
            # If the stem does not exist add and entry for it to the dicitonary.
            self.lemma_dict[key] = [value]
        elif not value in self.lemma_dict[key]:
            # If the stem exists append to the list of matched tokens.
            self.lemma_dict[key].append(value)

    def get(self, key):
        """ Returns the list of tokens matched to a stem. """
        assert len(self.lemma_dict) > 0
        assert key in self.lemma_dict
        return self.lemma_dict[key]

    def stem2lemma(self, key):
        """ Takes a stem as argument and returns a noun that has it as root.

            Args:
                key (str): The stem that will be converted to a lemma.

            Returns:
                lemma (str): A noun lemma that has the given stem as root.
        """

        noun_suffixes = ['ς', 'η', 'o', 'οι', 'α']
        tokens = self.get(key)
        for suffix in noun_suffixes:
            for token in tokens:
                matched_tokens = []
                if normalize(token).endswith(suffix):
                    matched_tokens.append(token)
            if len(matched_tokens) > 0:
                token_len = [len(token) for token in matched_tokens]
                return matched_tokens[token_len.index(min(token_len))]
        return tokens[0]

    def lemmatize(self, word):
        """ Converts a word to its noun form.

            Args:
                word (str): The word to be lemmatized.

            Returns:
                lemma (str): A noun version of the given word.
        """

        key = stem(word)[0]
        lemma = self.stem2lemma(key)
        return lemma
