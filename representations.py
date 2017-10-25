import os, sys
import joblib
import numpy as np

def get_cluster_reps(kmodel, dist_space):
    """ Represent the clusters of a K-means clustering model with the most
        important words of the three documents closest to the cluster center
        using the Tf-Idf matrix of the collection.
        
        Args:
            kmodel (obj): An sklearn K-means model.

        Returns: 
            cluster_reps (list(str)): The representations of the clusters.
            
    """
    cluster_reps = []
    if os.path.exists('test_data/dictionary.txt'):
        dictionary = joblib.load('test_data/dictionary.txt')
    else:
        print('No dictionary file found.')
        sys.exit(0)

    if os.path.exists('test_data/tfidf_sparse.txt'):
        tfidf = joblib.load('test_data/tfidf_sparse.txt')
    else:
        print('No topic model file found.')
        sys.exit(0)

    if os.path.exists('test_data/topic_model.txt'):
        topic_model = joblib.load('test_data/topic_model.txt')
    else:
        print('No topic model file found.')
        sys.exit(0)

    for cluster_id, cluster_center in enumerate(kmodel.cluster_centers_):
        # Find the three documents nearest to the cluster center.
        dist_vector = dist_space[:, cluster_id].argsort()
        nearest_doc_ids = dist_vector[:3]

        # Find the best term of each ldocument and add it to the cluster representation.
        best_term_ids = []
        for doc_id in nearest_doc_ids:
            tfidf_vector_dense = tfidf.getrow(doc_id).todense()
            sorted_tfidf_vector = tfidf_vector_dense.argsort().tolist()[0][::-1] 
            best_term_ids.append(sorted_tfidf_vector[0])
        cluster_reps.append([dictionary[term_id] for term_id in best_term_ids])

    return cluster_reps



    
