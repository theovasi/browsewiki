# -*- coding: utf-8 -*-
import os
import unittest
from unittest import mock
from toolset.corpus import Corpus
import joblib


class CorpusTest(unittest.TestCase):

    def setUp(self):
        self.filepaths = []
        for i in range(3):
            with open('file{}'.format(i), 'w+') as file_:
                file_.write('This is a test string.')
            self.filepaths.append('file{}'.format(i))

        with open('file3', 'w+') as file_:
            file_.write('This is a longer test string.')
        self.filepaths.append('file3')

        joblib.dump(self.filepaths, 'filepaths')

        self.corpus = Corpus(self.filepaths)

    def test_corpus_document_generator(self):
        for doc in self.corpus.document_generator():
            self.assertTrue(doc == 'This is a test string.'
                            or doc == 'This is a longer test string.')

    def test_corpus_stats(self):
        stat_log = self.corpus.stats()
        self.assertEqual(stat_log['n_docs'], 4)
        # Dot is counterd a word.
        self.assertEqual(stat_log['max_doc_size'], 7)
        self.assertEqual(stat_log['min_doc_size'], 6)
        self.assertEqual(stat_log['avg_doc_size'], 6.2)

    def tearDown(self):
        for i in range(4):
            os.remove('file{}'.format(i))
        os.remove('filepaths')


if __name__ == '__main__':
    unittest.main()
