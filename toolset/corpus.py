# !/usr/bin/python
# -*- coding: utf-8 -*-
""" Provides processing utilities for collections of text documents."""
import os
import time
import math
import joblib
from nltk import sent_tokenize, word_tokenize


class Corpus:
    """ Enables a number of operations on a collection of documents.

    A Corpus object must be initialized on a folder where each document can be
    found with the relative link external_folder/internal_folder/document_file.
    Every document file must contain only one document.

    Attributes:
        filepath_dict_path (str): Path to the pickled dictionary that matches
            filepaths to document ids. Allows processing the text documents in
            a specific order.

    """

    def __init__(self, filepaths):
        self.document_paths = filepaths

    def document_generator(self):
        """ Enables iterating over the documents of the formatted collection.

        Yields:
            The documents of the collection one by one in String form.

        """
        for i in range(len(self.document_paths)):
            path = self.document_paths[i]
            with open(path) as document_file:
                yield document_file.read()

    def stats(self, verbose=False):
        """ Calculate statistics about the collection.

        Calculates the number of documents in the collection as well as the
        maximum, minimum and average document size.

        Args:
            verbose (boolean): When set to true, the method will print
                the information during the calculation.

        Returns:
            stat_log (:obj: `dict`): Dictionary containing the results of the
                calculation.

        """
        start_time = time.time()
        n_docs, total_doc_size, max_doc_size, min_doc_size = 0, 0, 0, math.inf

        for doc in self.document_generator():
            n_docs += 1
            current_doc_size = 0
            tokens = [word for sent in sent_tokenize(doc)
                      for word in word_tokenize(sent)]
            current_doc_size = len(tokens)
            total_doc_size += current_doc_size

            max_doc_size = max(max_doc_size, current_doc_size)
            min_doc_size = min(min_doc_size, current_doc_size)

            if verbose and n_docs % 2000 == 0:
                print('Documents processed: ' + str(n_docs) + ', Rate: ' + str(
                    round(n_docs / (time.time() - start_time))) + 'docs/sec')

        stat_log = dict()
        stat_log['n_docs'] = n_docs
        stat_log['max_doc_size'] = max_doc_size
        stat_log['min_doc_size'] = min_doc_size
        stat_log['avg_doc_size'] = round(total_doc_size / n_docs, 1)

        if verbose:
            print()
            print()
            print('Number of documents: ' + str(n_docs))
            print()
            print('Document size metrics: ')
            print('\tMax: ' + str(max_doc_size))
            print('\tMin: ' + str(min_doc_size))
            print('\tAverage: ' + str(round(total_doc_size / n_docs, 1)))

        return stat_log
