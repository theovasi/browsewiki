import os, sys, logging, scipy, joblib
import math
import argparse

from toolset.corpus import Corpus
from gensim import corpora, models, matutils
from sklearn.cluster import MiniBatchKMeans as mbk
from toolset import mogreltk


def make_topicspace(data_file_path, stopwords_file_path=None,
                    n_topics=300, method='lda', n_clusters=8):
    # Allow gensim to print additional info while executing.
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    if not os.path.exists(data_file_path+'/formatted'):
        print('No corpus file found.')
    collection = Corpus(data_file_path+'/formatted',
                        filepath_dict_path=data_file_path+'/filepath_dict.txt')

    # First pass of the collection to create the dictionary.
    if not os.path.exists(data_file_path + '/dictionary.txt'):
        print('Generating dictionary...')
        dictionary = corpora.Dictionary()
        batch_size = 0
        max_batch_size = 2000
        batch = []

        for i, text in enumerate(collection.document_generator()):
            if stopwords_file_path is not None:
                batch.append(mogreltk.stem(text, stopwords_file_path))
            else:
                batch.append(mogreltk.stem(text))
            batch_size += 1
            if batch_size >= max_batch_size:
                dictionary.add_documents(batch, prune_at=5000)
                batch_size = 0
                batch = []
        dictionary.add_documents(batch, prune_at=5000)
        dictionary.filter_extremes(no_below=100, no_above=0.15)
        joblib.dump(dictionary, data_file_path + '/dictionary.txt')

    # Second pass of the collection to generate the bag of words representation.
    if not os.path.exists(data_file_path + '/corpus.txt'):
        print('Generating corpus...')
        if not 'dictionary' in locals():
            dictionary = joblib.load(data_file_path + '/dictionary.txt')
        if stopwords_file_path is not None:
            corpus = [dictionary.doc2bow(mogreltk.stem(text, stopwords_file_path))
                      for text in collection.document_generator()]
        else:
            corpus = [dictionary.doc2bow(mogreltk.stem(text))
                      for text in collection.document_generator()]
        joblib.dump(corpus, data_file_path + '/corpus.txt')

    # Transform from BoW representation to tf-idf.
    if not os.path.exists(data_file_path + '/tfidf_model.txt'):
        print('Generating tf-idf matrix...')
        if not 'corpus' in locals():
            corpus = joblib.load(data_file_path + '/corpus.txt')
        tfidf = models.TfidfModel(corpus)
        joblib.dump(tfidf, data_file_path + '/tfidf_model.txt')
        corpus_tfidf = tfidf[corpus]
        tfidf_sparse = matutils.corpus2csc(corpus_tfidf)
        tfidf_sparse = scipy.sparse.csc_matrix.transpose(tfidf_sparse).tocsr()
        joblib.dump(tfidf_sparse, data_file_path + '/tfidf_sparse.txt')

    # Apply Latent Dirichlet Allocation.
    if not os.path.exists(data_file_path + '/topic_model.txt'):
        if not 'dictionary' in locals():
            dictionary = joblib.load(data_file_path + '/dictionary.txt')
        if not 'corpus' in locals():
            corpus = joblib.load(data_file_path + '/corpus.txt')
        if not 'tfidf' in locals():
            tfidf = joblib.load(data_file_path + '/tfidf_model.txt')
            corpus_tfidf = tfidf[corpus]
        if method == 'lsa':
            print('Applying Latent Semantic Analysis for {} topics.'.format(n_topics))
            lsa = models.lsimodel.LsiModel(corpus=corpus_tfidf, id2word=dictionary,
                                           num_topics=n_topics)
            joblib.dump(lsa, data_file_path + '/topic_model.txt')
            transformed_corpus = lsa[corpus]
        else:
            print('Applying Latent Dirichlet Allocation for {} topics.'.format(n_topics))
            lda = models.ldamodel.LdaModel(corpus=corpus_tfidf, id2word=dictionary,
                                           num_topics=n_topics, passes=2)
            joblib.dump(lda, data_file_path + '/topic_model.txt')
            transformed_corpus = lda[corpus]

        # Convert topic space matrix to sparse in the Compressed Sparse Row format.
        topic_space = matutils.corpus2csc(transformed_corpus)

        # Transpose the topic space matrix because it will be used with sklearn and
        # it needs the documents in the rows.
        topic_space = scipy.sparse.csc_matrix.transpose(topic_space).tocsr()
        joblib.dump(topic_space, data_file_path + '/topic_space.txt')

    # Apply clustering using KMeans
    if not os.path.exists(data_file_path + '/kmodel.txt'):
        if not 'topic_space' in locals():
            topic_space = joblib.load(data_file_path + '/topic_space.txt')
        kmodel = mbk(n_clusters=n_clusters, n_init=10, reassignment_ratio=0.03, verbose=True)
        kmodel.fit(topic_space)
        dist_space = kmodel.transform(topic_space)
        joblib.dump(kmodel, data_file_path + '/kmodel.txt')
        joblib.dump(dist_space, data_file_path + '/dist_space.txt')

    if not os.path.exists('{}/lemmatizer.txt'.format(data_file_path)):
        lemmatizer = mogreltk.Lemmatizer()
        lemmatizer.fit(collection.document_generator(), stopwords_file_path)
        joblib.dump(lemmatizer, '{}/lemmatizer.txt'.format(data_file_path))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process input filepath.')
    parser.add_argument('data_file_path', type=str,
                        help='The path to the data directory.')
    parser.add_argument('-s', '--stop', type=str,
                        help='The path to the stopwords file.')
    parser.add_argument('-t', '--n_topics', type=int,
                        help='The number of topics that will be extracted.')
    parser.add_argument('-m', '--method', type=str,
                        help='The topic modeling method to be used.')
    parser.add_argument('-k', '--n_clusters', type=int ,
                        help='The number of clusters to be created.')
    args = parser.parse_args()
    make_topicspace(data_file_path=args.data_file_path, stopwords_file_path=args.stop,
                    n_topics=args.n_topics, method=args.method, n_clusters=args.n_clusters)
