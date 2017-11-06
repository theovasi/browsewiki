""" This module offers a number tools for processing modern Greek text. """
import re
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
    stem = GreekStemmer().stem
    if stopwords_file_path is not None:
        with open(stopwords_file_path) as file_data:
            stopwords = file_data.read().splitlines()
        stopwords = [normalize(stopword) for stopword in stopwords]

    tokens = [
        word.lower()
        for sent in sent_tokenize(text)
        for word in word_tokenize(sent)
    ]
    # Remove tokens that do not contain letters.
    for token in tokens:
        if not re.search('^\d*$', token)\
           and not (stopwords_file_path is not None and normalize(token) in stopwords)\
           and not re.search('[\.,!;:\"\'-«»\.\.\.%]', token):
            filtered_tokens.append(stem(normalize(token).upper()).lower())

    return filtered_tokens
