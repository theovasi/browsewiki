# !/usr/bin/python
# -*- coding: utf-8 -*-
""" This module offers tools for processing modern Greek text."""
import time
import math
import re
from operator import itemgetter
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
            filtered_tokens: The original text as a list of tokens excluding
            stopwords, punctuation and numbers.
    """
    filtered_tokens = []
    if stopwords_file_path is not None:
        with open(stopwords_file_path) as file_data:
            stopwords = file_data.read().splitlines()
        stopwords = [normalize(stopword) for stopword in stopwords]

    tokens = [
        normalize(word.lower())
        for sent in sent_tokenize(text)
        for word in word_tokenize(sent)
    ]
    # Remove tokens that 1) do not contain letters 2) are stopwords
    # 3) are punctuation 4) contain less than 3 letters.
    for token in tokens:
        if not re.search(r'^\d*$', token)\
                and not (stopwords_file_path is not None
                         and normalize(token) in stopwords)\
                and not re.search(r'[\.,!;:\"\'-«»\.\.\.%]', token)\
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
    stems = [stemmer.stem(token.upper()).lower()
             for token in tokens]
    return stems


class Lemmatizer(object):
    """ Offers pseudo-lemmatization functionality for the Greek language.

    When the lemmatizer is fit on a text collection, it creates a dictionary
    that matches each unique stem to a list of tuples. Each tuple contains a
    token and its frequency of occurence in the collection. To lemmatize a
    word, the word is stemmed first and then the most frequent token that has
    been matches to the word's stem is selected as its lemma.

    """

    def __init__(self):
        self.lemma_dict = dict()

    def fit(self, text_collection_iterable,
            stopwords_file_path=None, verbose=False):
        """ Process a text collection and create a dictionary that matches
            stems to tokens.

            Args:
                text_collection_iterable (obj): A list, generator or other
                    iterable object that returns documents in string format.
                stopwords_file_path (str): The path to a file containing
                    stopwords. Stopwords are ignored during the processing of
                    the collection.

        """
        if verbose:
            print('Fitting lemmatizer...')
        n_docs = 0

        start_time = time.time()
        for doc in text_collection_iterable:
            words_tokenized = tokenize(doc, stopwords_file_path)
            words_stemmed = stem(doc, stopwords_file_path)
            for index, word in enumerate(words_stemmed):
                self.add(word, words_tokenized[index])

            n_docs += 1
            if verbose and n_docs % 2000 == 0:
                print('Processed {} documents... - {} docs/s'.format
                      (n_docs,
                       math.floor(n_docs / (time.time() - start_time))))

    def add(self, stem_, token):
        """ Add a matching of a token to a stem."""
        if stem_ not in self.lemma_dict:
            self.lemma_dict[stem_] = [[token, 0]]
        else:
            token_list = [token_entry[0]
                          for token_entry in self.lemma_dict[stem_]]
            if token not in token_list:
                self.lemma_dict[stem_].append([token, 0])
            else:
                for token_entry in self.lemma_dict[stem_]:
                    if token_entry[0] == token:
                        token_entry[1] += 1

    def get(self, key):
        """ Returns the list of tokens matched to a stem. """
        return self.lemma_dict[key]

    def stem2lemma(self, key):
        """ Takes a stem as argument and returns a token that has it as root.

            Args:
                key (str): The stem that will be converted to a lemma.

            Returns:
                (str): The most frequent token matched to a specific stem

        """

        tokens = self.get(key)
        # Return the most frequent token.
        return max(tokens, key=itemgetter(1))[0]  # Faster than lambda.

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
