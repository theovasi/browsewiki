import os, sys
import joblib
import numpy as np
from toolset import mogreltk

def get_common_terms(tfidf_vectors, depth=10):
    """ Find common terms between documents by looking at their top X most important ones.

        Based on the depth argument, looks at the top X most important terms based on the
        Tf-Idf representation of the documents and returns those that all the documents
        have in common.

        Args:
            document_vectors (list): A list of document Tf-Idf representations.
            depth (int): The number of terms with the highest Tf-Idf value that will be considered.

        Returns:
            common_terms (list): The list of common important terms.
    """
    # Keep only the top X highest scoring terms.
    sorted_tfidf_vectors = [vector.argsort().tolist()[0][::-1][:depth] for vector in tfidf_vectors]

    common_terms = []
    for term_id in sorted_tfidf_vectors[0]:
        is_common = True
        # Check if the term id exists in all the other sorted vectors.
        for index in range(1, len(sorted_tfidf_vectors)):
            if term_id not in sorted_tfidf_vectors[index]:
                is_common = False
                break
        # If a common term id add it to the common list.
        if is_common:
            common_terms.append(term_id)

    return common_terms

def get_cluster_reps(kmodel, dist_space, data_file_path, depth):
    """ Represent the clusters of a K-means clustering model with the most
        important words of the three documents closest to the cluster center
        using the Tf-Idf matrix of the collection.

        Args:
            kmodel (obj): An sklearn K-means model.

        Returns:
            cluster_reps (list(str)): The representations of the clusters.

    """
    cluster_reps = []
    if os.path.exists('{}/dictionary.txt'.format(data_file_path)):
        dictionary = joblib.load('{}/dictionary.txt'.format(data_file_path))
    else:
        print('No dictionary file found.')
        sys.exit(0)

    if os.path.exists('{}/tfidf_sparse.txt'.format(data_file_path)):
        tfidf = joblib.load('{}/tfidf_sparse.txt'.format(data_file_path))
    else:
        print('No topic model file found.')
        sys.exit(0)

    if os.path.exists('{}/topic_model.txt'.format(data_file_path)):
        topic_model = joblib.load('{}/topic_model.txt'.format(data_file_path))
    else:
        print('No topic model file found.')
        sys.exit(0)

    if os.path.exists('{}/lemmatizer.txt'.format(data_file_path)):
        lemmatizer = joblib.load('{}/lemmatizer.txt'.format(data_file_path))
    else:
        print('No lemmatizer file found.')
        sys.exit(0)

    for cluster_id, cluster_center in enumerate(kmodel.cluster_centers_):
        # Find the three documents nearest to the cluster center.
        dist_vector = dist_space[:, cluster_id].argsort()
        nearest_doc_ids = dist_vector[:10]

        # Find the best term of each ldocument and add it to the cluster
        # representation.
        tfidf_vectors = []
        for doc_id in nearest_doc_ids:
            tfidf_vector_dense = tfidf.getrow(doc_id).todense()
            tfidf_vectors.append(tfidf_vector_dense)
        common_terms = [lemmatizer.stem2lemma(dictionary[term])
                        for term in get_common_terms(tfidf_vectors, depth)]
        cluster_reps.append(common_terms)

    return cluster_reps
