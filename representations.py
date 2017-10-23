import os, sys
import joblib

def get_cluster_reps(kmodel):
    """ Represent the clusters of a K-means clustering model using their most
        important words using the Tf-Idf matrix of the collection.
        
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
        print('No tfidf file found.')
        sys.exit(0)

    if os.path.exists('test_data/topic_model.txt'):
        topic_model = joblib.load('test_data/topic_model.txt')
    else:
        print('No topic model file found.')
        sys.exit(0)

    for cluster_center in kmodel.cluster_centers_:
        sorted_cluster_center = cluster_center.argsort()[::-1]
        #  cluster_reps.append([dictionary[sorted_cluster_center[i]] for i in range(3)])
        cluster_reps.append([dictionary[term_tuple[0]]
                            for term_tuple in topic_model.get_topic_terms(
                                sorted_cluster_center[0], topn=3)])
    return cluster_reps



    
