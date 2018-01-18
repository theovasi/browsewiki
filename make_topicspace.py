import os
import sys
import logging
import scipy
import joblib
import math
import argparse

from toolset.corpus import Corpus
from gensim import corpora, models, matutils
from sklearn.cluster import MiniBatchKMeans as mbk
from toolset import mogreltk
from sklearn.neighbors import NearestNeighbors as nn
from toolset.cluster_metrics import cluster_metrics
from toolset.visualize import get_cluster_reps 


def make_topicspace(data_file_path, stopwords_file_path=None,
                    n_topics=300, method='lda', n_clusters=8):
    # Allow gensim to print additional info while executing.
    logging.basicConfig(
        format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    corpus_frame = joblib.load('{}/corpus_frame.txt'.format(data_file_path))
    filepaths = list(corpus_frame['filepath'])
    collection = Corpus(filepaths)
    print('-- Loaded corpus')

    # First pass of the collection to create the dictionary.
    if not os.path.exists(data_file_path + '/dictionary.txt'):
        print('-- Generating dictionary')
        dictionary = corpora.Dictionary()
        batch_size = 0
        max_batch_size = 2000
        batch = []
        if stopwords_file_path is not None:
            with open(stopwords_file_path) as stopwords_file:
                stopwords = stopwords_file.read().splitlines()

        for i, text in enumerate(collection.document_generator()):
            if stopwords is not None:
                batch.append(mogreltk.stem(text, stopwords))
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
        if 'dictionary' not in locals():
            dictionary = joblib.load(data_file_path + '/dictionary.txt')
            if stopwords_file_path is not None:
                with open(stopwords_file_path) as stopwords_file:
                    stopwords = stopwords_file.read().splitlines()
            print('-- Loaded dictionary')
        print('-- Generating corpus')
        if stopwords is not None:
            corpus = [dictionary.doc2bow(mogreltk.stem(text, stopwords))
                      for text in collection.document_generator()]
        else:
            corpus = [dictionary.doc2bow(mogreltk.stem(text))
                      for text in collection.document_generator()]
        joblib.dump(corpus, data_file_path + '/corpus.txt')

    # Transform from BoW representation to tf-idf.
    if not os.path.exists(data_file_path + '/tfidf_model.txt'):
        if not 'corpus' in locals():
            corpus = joblib.load(data_file_path + '/corpus.txt')
            print('-- Loaded corpus')
        print('-- Generating tf-idf matrix')
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
            print('-- Loaded dictionary')
        if not 'corpus' in locals():
            corpus = joblib.load(data_file_path + '/corpus.txt')
            print('-- Loaded corpus')
        if not 'tfidf' in locals():
            tfidf = joblib.load(data_file_path + '/tfidf_model.txt')
            print('-- Loaded tfidf model')
            corpus_tfidf = tfidf[corpus]
        if method == 'lsa':
            print('-- Applying Latent Semantic Analysis for {} topics'.format(n_topics))
            lsa = models.lsimodel.LsiModel(corpus=corpus_tfidf, id2word=dictionary,
                                           num_topics=n_topics)
            joblib.dump(lsa, data_file_path + '/topic_model.txt')
            transformed_corpus = lsa[corpus]
        else:
            print(
                '-- Applying Latent Dirichlet Allocation for {} topics'.format(n_topics))
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
            print('-- Loaded topic space matrix')
        best_silhouette_score = -1
        best_kmodel = None
        for index in range(100):
            kmodel = mbk(n_clusters=n_clusters, n_init=100,
                         reassignment_ratio=0.03)
            kmodel.fit(topic_space)
            silhouette_score = cluster_metrics(kmodel, topic_space)
            if best_silhouette_score < silhouette_score:
                best_silhouette_score = silhouette_score
                best_kmodel = kmodel
        dist_space = kmodel.transform(topic_space)
        print('Picked K-means model with silhouette score: {}'.format(
              best_silhouette_score))
        joblib.dump(best_kmodel, data_file_path + '/kmodel.txt')
        joblib.dump(dist_space, data_file_path + '/dist_space.txt')

    if not os.path.exists('{}/lemmatizer.txt'.format(data_file_path)):
        lemmatizer = mogreltk.Lemmatizer()
        lemmatizer.fit(collection.document_generator(),
                       stopwords_file_path, True)
        joblib.dump(lemmatizer, '{}/lemmatizer.txt'.format(data_file_path))

    # Generate cluster labels.
    if not os.path.exists('{}/cluster_reps.txt'.format(data_file_path)):
        if not 'tfidf_sparse' in locals():
            tfidf_sparse = joblib.load(data_file_path + '/tfidf_sparse.txt')
            print('-- Loaded tfidf matrix.')
        if not 'best_kmodel' in locals():
            best_kmodel = joblib.load(data_file_path + '/kmodel.txt')
            print('-- Loaded K-means model.')
        if not 'dictionary' in locals():
            dictionary = joblib.load(data_file_path + '/dictionary.txt')
            print('-- Loaded dictionary.')
        if not 'lemmatizer' in locals():
            lemmatizer = joblib.load(data_file_path + '/lemmatizer.txt')
            print('-- Loaded lemmatizer.')
        cluster_reps = get_cluster_reps(tfidf_sparse, best_kmodel, dictionary, lemmatizer)
        joblib.dump(cluster_reps, '{}/cluster_reps.txt'.format(data_file_path))

    if not os.path.exists('{}/nn_model.txt'.format(data_file_path)):
        if 'tfidf_sparse' not in locals():
            tfidf_sparse = joblib.load('{}/tfidf_sparse.txt'.format(data_file_path))
        nn_model = nn(n_neighbors=1000, radius=10)
        nn_model.fit(tfidf_sparse)
        joblib.dump(nn_model, '{}/nn_model.txt'.format(data_file_path))



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
    parser.add_argument('-k', '--n_clusters', type=int,
                        help='The number of clusters to be created.')
    args = parser.parse_args()
    make_topicspace(data_file_path=args.data_file_path, stopwords_file_path=args.stop,
                    n_topics=args.n_topics, method=args.method, n_clusters=args.n_clusters)
