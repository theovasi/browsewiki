#!/usr/bin/python3.6
"""Setup.

Usage:
    setup.py <language>

Options:
    -h --help    Show this page.
"""
import os
import shutil

import re
import nltk
import glob
from docopt import docopt
from urllib import request
from bs4 import BeautifulSoup as bs
from toolset.wiki_format import wiki_format
from toolset.make_topicspace import make_topicspace
nltk.download('punkt')


def app_setup(language):
    # Build the link to download the Wikipedia dump.
    mirror = 'ftp.acc.umu.se/mirror/wikimedia.org/dumps'
    dump_names = {
        'english': 'enwiki',
        'greek': 'elwiki'
    }
    dump_name = dump_names[language]

    # There's a page in the mirror where all the dump versions are listed
    # from where we determing the latest one.
    version_select_page = request.urlopen(
        '/'.join(['http:/', mirror, dump_name]))
    parsed_version_select_page = bs(version_select_page, 'html5lib')

    available_versions = []
    for link in parsed_version_select_page.findAll('a'):
        result = re.search(r"[0-9]+", str(link))
        if result:
            available_versions.append(result.group(0))

    latest_version = available_versions[-1]
    available_versions = available_versions[:-1]
    while not glob.glob(r'data/' + dump_name + '-*-pages-articles.xml.bz2'):
        latest_version = available_versions[-1]
        available_versions = available_versions[:-1]

        # Now the download link can be built.
        if glob.glob(r'data/' + dump_name + '-.*-pages-articles.xml.bz2'):
            print('Latest dump version already exists.')
        else:
            if os.path.exists('data'):
                shutil.rmtree('data')
            os.mkdir('data')
            os.chdir('data')
            os.system('wget --no-check-certificate '
                      + '/'.join(['https:/',
                                  mirror, dump_name, latest_version,
                                  (dump_name+'-' + latest_version +
                                   '-pages-articles.xml.bz2')]))
            os.chdir('../')

    dump_version = re.match(r'data/' + dump_name + '-(.*)-pages-articles.xml.bz2', glob.glob(r'data/' + dump_name + '-*-pages-articles.xml.bz2')[0]).group(1)

    # Extract the documents form the compressed XML with Wikiextractor.
    # Wikiextractor puts the documents in txt files of equal size.
    if not os.path.exists('data/' + dump_name + '-' + dump_version):
        os.system('git clone https://github.com/attardi/wikiextractor')
        os.mkdir('data/' + dump_name + '-' + dump_version)
        os.chdir('wikiextractor')
        os.system('python3 -m wikiextractor.WikiExtractor '
                  '--no-templates -o ../data/' + dump_name + '-' +
                  dump_version + ' ../data/' + dump_name + '-' +
                  dump_version + '-pages-articles.xml.bz2')
        os.chdir('../')
        shutil.rmtree('wikiextractor')
    else:
        print('Latest dump version already extracted.')

    # Format the documents so that there is one document per txt file and
    # also create a dataframe that matches the document's filepaths to their
    # titles. short summaries and web links.
    wiki_format('data/' + dump_name + '-' + dump_version, sub_size=500000,
                output_file_path='appdata')

    # Generate the topic space and build all the models the app uses.
    make_topicspace('appdata', 'assets/el-stopwords.txt')


if __name__ == '__main__':
    args = docopt(__doc__)
    app_setup(args['<language>'])
