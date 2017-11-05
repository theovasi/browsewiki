# -*- coding: utf-8 -*-
""" Allows processing and analysis of a collection of text documents.
"""
import os, shutil, time
import itertools, joblib
import pandas as pd


class Corpus(object):
    """ Enables a number of operations on a collection of documents.

    A Corpus object must be initialized on a folder where each document can be
    found with the relative link external_folder/internal_folder/document_file.
    Every document file must contain only one document.
    Attributes:
        corpus_file_path (str): The path to the folder of the document collection.
    """

    def __init__(self, corpus_file_path, filepath_dict_path):
        self.corpus_file_path = corpus_file_path
        #
        self.document_paths = joblib.load(filepath_dict_path)

    def document_generator(self):
        """ Enables iterating over the documents of the formatted collection.

        Yields:
            The documents of the collection one by one in String form.

        """
        for i in range(len(self.document_paths)):
            path = self.document_paths[i]
            with open(path) as document_file:
                yield document_file.read()

    def get_vocabulary(self):
        """ Get the vocabulary of the document collection.

        Returns:
            pandas.Series: Matching of stems and the tokens they derived from.
        """
        vocabulary_tokenized = []
        vocabulary_stemmed = []
        # Initiate a new document generator when this method is called.
        corpus = self.document_generator()

        for document in corpus:
            document_tokens = tokenize(document)
            vocabulary_tokenized.extend(document_tokens)
            # Remove duplicate tokens by casting to set and back to list.
            vocabulary_tokenized = list(set(vocabulary_tokenized))
        vocabulary_stemmed = stem(vocabulary_tokenized)

        # Create pandas series that matched stems to tokens with stems as indeces.
        vocabulary = pd.Series(vocabulary_tokenized, index=vocabulary_stemmed)

        return vocabulary

    def get_stats(self):
        """ Prints statistics about the collection.

        Calculates and prints the number of documents in the collection as well as
        the maximum, the minimun and the average document size.
        """
        start_time = time.time()
        n_docs = 0
        # Document size metrics expressed in number of features.
        doc_size = 0
        total_doc_size = 0
        max_doc_size = 0
        min_doc_size = 0
        avg_doc_size = 0
        for path in self.document_paths:
            with open(path) as document_file_content:
                doc_size = 0
                for line in document_file_content:
                    if not (line.startswith('<doc') or
                            line.startswith('</doc>')):
                        doc_size += len(line.split(' '))
                # Update metric values
                if n_docs == 0:
                    min_doc_size = doc_size
                n_docs += 1
                print('Documents processed: ' + str(n_docs) + ', Rate: ' + str(
                    round(n_docs / (time.time() - start_time))) + 'docs/sec')
                total_doc_size += doc_size
                if doc_size > max_doc_size:
                    max_doc_size = doc_size
                if doc_size < min_doc_size:
                    min_doc_size = doc_size
        avg_doc_size = total_doc_size / n_docs
        print()
        print()
        print('Number of documents: ' + str(n_docs))
        print()
        print('Document size metrics: ')
        print('    Max: ' + str(max_doc_size))
        print('    Min: ' + str(min_doc_size))
        print('    Average: ' + str(avg_doc_size))

    def get_title(self, docid):
        """ Get the title of a document by id.

        Each document's id depends on the order in which it got processed. For example,
        the document with id=5 was the fifth document to be processed.

        Returns:
            title (str): The title of the document.

        """

        with open(self.document_paths[docid]) as document_file:
            return document_file.readlines()[1].strip()
