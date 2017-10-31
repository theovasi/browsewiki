import os
import itertools
import math
import random
import shutil
import time
import xml.etree.ElementTree as ET
from xml.sax.saxutils import escape
import joblib
import argparse
import re


def wiki_format(text_file_path,
                ignore_list_path=None,
                sub_size=None,
                output_file_path=os.getcwd()):
    """ Formats the collection so that there is one document per file.

        Before this method is run the collection is in the format generated by extracting
        the documents from a WikiPedia XML dump using Wikiextractor
        (github.com/attardi/wikiextractor).

        Arguments:
            sub_size (int, optional): Specifies the number of documents that will consist the
                formatted collection. Defaults to None and in this case the all the documents
                are kept during the formatting.
            output_file_path (str): The path to the file where the formatted collection will
                be saved. By default creates a file named 'formatted' in the current working
                directory.

        """

    # If a formatted collection directory already exists, overwrite it.
    if os.path.exists(output_file_path):
        shutil.rmtree(output_file_path)
        os.makedirs(output_file_path)

    n_docs = 0
    n_ignored_docs = 0
    title_dict = dict()
    link_dict = dict()
    summary_dict = dict()
    filepath_dict = dict()
    if ignore_list_path is not None:
        ignore_list = joblib.load(ignore_list_path)
    start_time = time.time()

    for document_folder in os.listdir(text_file_path):
        os.makedirs(output_file_path + '/formatted/'+ document_folder)
        for document_file in os.listdir(text_file_path + '/' +
                                        document_folder):
            # The document's XML like format does not have a root element so it
            # needs to be added in order for the ElementTree to be created.
            with open(text_file_path + '/' + document_folder + '/' +
                      document_file) as document_content:
                # Escape all lines except <doc> tag lines to avoid XML parsing
                # errors
                document_content_escaped = []
                for line in document_content.readlines():
                    if (not line.startswith('<doc id') and
                            not line.startswith('</doc>')):
                        document_content_escaped.append(escape(line))
                    else:
                        document_content_escaped.append(line)

                document_file_iterator = \
                        itertools.chain('<root>',
                                        document_content_escaped, '</root>')
                # Parse the document file using the iterable.
                documents = ET.fromstringlist(document_file_iterator)
                # Each document file contains multiple documents each
                # wrapped in a doc tag.
                for i, doc in enumerate(documents.findall("doc")):
                    # Pick documents at random from the whole collection.
                    if sub_size != None:
                        pos = float(sub_size) / pow(10, 4)
                        if random.random() > pos:
                            continue
                    doc_text = doc.text.splitlines()
                    title = doc_text[1]
                    # Skip document if it is in the ignore list.
                    if ignore_list_path is not None and title in ignore_list:
                        n_ignored_docs += 1
                        continue

                    # Skip document if its title contains only numbers.
                    if re.search('^\d*$', title.replace(' ', '')):
                        n_ignored_docs += 1
                        continue

                    link = title.replace(' ', '_')
                    summary = doc_text[3][:160]
                    # Save each document in a separate file.
                    filepath = '/'.join([
                        output_file_path, 'formatted', document_folder,
                        document_file + '_' + str(i)
                    ])

                    title_dict[n_docs] = title
                    link_dict[n_docs] = link
                    summary_dict[n_docs] = summary
                    filepath_dict[n_docs] = filepath

                    with open(filepath, 'wb+') as output_document_file:
                        output_document_file.write(doc.text.encode('utf-8'))

                    # If a subcollection size has been specified, stop when it
                    # is reached.
                    n_docs += 1
                    if sub_size is not None:
                        print('Picked {}/{} documents at random, {} docs/s'.format(n_docs, sub_size,
                              math.floor(n_docs / (time.time() - start_time))))
                    elif n_docs%10000 == 0:
                        print('Added {} documents to the collection, {} docs/s'.format(n_docs,
                              math.floor(n_docs / (time.time() - start_time))))

                    if sub_size != None and n_docs >= sub_size:
                        joblib.dump(title_dict,
                                    output_file_path + '/title_dict.txt')
                        joblib.dump(link_dict,
                                    output_file_path + '/link_dict.txt')
                        joblib.dump(summary_dict,
                                    output_file_path + '/summary_dict.txt')
                        joblib.dump(filepath_dict,
                                    output_file_path + '/filepath_dict.txt')
                        return 0
    joblib.dump(title_dict, output_file_path + '/title_dict.txt')
    joblib.dump(link_dict, output_file_path + '/link_dict.txt')
    joblib.dump(summary_dict, output_file_path + '/summary_dict.txt')
    joblib.dump(filepath_dict, output_file_path + '/filepath_dict.txt')
    print('{} documents processed , {} added to the collection, {} ignored.'.format(n_docs+n_ignored_docs,
          n_docs, n_ignored_docs))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Process input and output filepaths.')
    parser.add_argument(
        'text_file_path', type=str,
        help='The path to the text file that WikiExtractor created.')
    parser.add_argument('-s', '--size', type=int,
                        help='The size of the subcollection.')
    parser.add_argument('-i', '--ignore', type=str,
                        help="""Path to title list of documents that won't be
                        included in the formatted collection.""")
    parser.add_argument('-o', '--output', type=str,
                        help='The path where the output files will be saved.')
    args = parser.parse_args()
    wiki_format(args.text_file_path, args.ignore, args.size, args.output)
