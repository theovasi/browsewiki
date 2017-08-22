import os, sys, joblib, logging, scipy
import math
import argparse

from toolset.corpus import Corpus as crp
from tmp_gen import tmp_gen
from toolset.corpus import tokenize
from gensim import corpora, models, matutils
from sklearn.cluster import MiniBatchKMeans as mbk


def make_topicspace(data_file_path):
    # Allow gensim to print additional info while executing.
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    # Tranform the collection from the format wikiextractor produces to a one
    # document per file format.
    if os.path.exists(data_file_path + '/formatted'):
        corpus = crp(data_file_path + '/formatted')
    else:
        print('Error: Text collection directory not found.')
        sys.exit(0)

    # First pass of the collection to create the dictionary.
    if not os.path.exists(data_file_path + '/dictionary.txt'):
        print('Generating dictionary...')
        dictionary = corpora.Dictionary()
        batch_size = 0
        max_batch_size = 20000
        batch = []

        for i, text in enumerate(tmp_gen()):
            batch.append(tokenize(text))
            batch_size += 1
            if batch_size >= max_batch_size:
                dictionary.add_documents(batch, prune_at=10000)
                batch_size = 0
                batch = []
        dictionary.add_documents(batch, prune_at=10000)
        dictionary.filter_extremes(no_below=50, no_above=0.15)
        joblib.dump(dictionary, data_file_path + '/dictionary.txt')

    # Second pass of the collection to generate the bag of words representation.
    if not os.path.exists(data_file_path + '/corpus.txt'):
        print('Generating corpus...')
        if not 'dictionary' in locals():
            dictionary = joblib.load(data_file_path + '/dictionary.txt')
        corpus = [dictionary.doc2bow(tokenize(text))
                  for text in tmp_gen()]
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
    if not os.path.exists(data_file_path + '/lda_model.txt'):
        print('Applying Latent Dirichlet Allocation')
        if not 'dictionary' in locals():
            dictionary = joblib.load(data_file_path + '/dictionary.txt')
        if not 'corpus' in locals():
            corpus = joblib.load(data_file_path + '/corpus.txt')
        if not 'tfidf' in locals():
            tfidf = joblib.load(data_file_path + '/tfidf_model.txt')
            corpus_tfidf = tfidf[corpus]
        lda = models.ldamodel.LdaModel(corpus=corpus_tfidf, id2word=dictionary,
                                       num_topics=300, passes=2)
        joblib.dump(lda, data_file_path + '/lda_model.txt')
        corpus_lda = lda[corpus]
        # Convert topic space matrix to sparse in the Compressed Sparse Row format.
        topic_space = matutils.corpus2csc(corpus_lda)
        # Transpose the topic space matrix because it will be used with sklearn and
        # it needs the documents in the rows.
        topic_space = scipy.sparse.csc_matrix.transpose(topic_space).tocsr()
        joblib.dump(topic_space, data_file_path + '/topic_space.txt')

    # Apply clustering using KMeans
    if not 'topic_space' in locals():
        topic_space = joblib.load(data_file_path + '/topic_space.txt')
    kmodel = mbk(n_clusters=12, verbose=True)
    kmodel.fit(topic_space)
    dist_space = kmodel.transform(topic_space)
    joblib.dump(kmodel, data_file_path + '/kmodel.txt')
    joblib.dump(dist_space, data_file_path + '/dist_space.txt')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process input and output filepaths.')
    parser.add_argument('data_file_path', type=str,
                        help='The path to the data directory.')
    args = parser.parse_args()
    make_topicspace(args.data_file_path)
