# -*- coding: utf-8 -*-
import unittest
from unittest import mock
from toolset import corpus


class CorpusTest(unittest.TestCase):

    @mock.patch('joblib.load')
    def test_raises_exception_on_invalid_document_dir(self, mock_load):
        mock_load.return_value = ''
        self.assertRaises(FileNotFoundError,
                          corpus.Corpus, 'nodir', 'nofile')


if __name__ == '__main__':
    unittest.main()
