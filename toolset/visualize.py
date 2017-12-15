import os
import sys
import joblib
import numpy as np
from toolset import mogreltk
from collections import Counter


def top_terms(tfidf_vectors, depth=10, top_n=3):
    """ Find terms with the highest cumulative Tf-Idf score across all vectors.

        Args:
            document_vectors (list): A list of document Tf-Idf representations.
            depth (int): The number of terms with the highest Tf-Idf value that
                will be considered from each vector.
            top_n (int): The number of highest scoring terms to be returned.

        Returns:
            top_terms (list): The terms with the highest cumulative Tf-Idf score.
    """
    # A dictionary that matches term ids to cumulative Tf-Idf score.
    term_dict = dict()
    for vector in tfidf_vectors:
        for term_id in vector.argsort().tolist()[0][::-1][:depth]:
            if term_id not in term_dict:
                term_dict[term_id] = vector.tolist()[0][term_id]
            else:
                term_dict[term_id] += vector.tolist()[0][term_id]

    # Find the n terms with the highest cumulative Tf-Idf score.
    terms = list(term_dict.keys())
    tfidf_scores = list(term_dict.values())
    sorted_tfidf_scores = np.argsort(tfidf_scores).tolist()[::-1][:top_n]
    best_score_terms = [terms[index] for index in sorted_tfidf_scores]

    return best_score_terms


def get_cluster_reps(tfidf, kmodel, dist_space, data_file_path, depth):
    """ Represent clusters with their most important words using Tf-Idf.

        Args:
            kmodel (obj): An sklearn K-means model.
            dist_space (obj): Matrix of document distance from cluster centers.
            data_file_path (str): Path to the application data.
            depth (int): The number of documents closest to the each cluster
                center that will be considered.

        Returns:
            cluster_reps (list(str)): The representations of the clusters.

    """
    cluster_reps = []
    if os.path.exists('{}/dictionary.txt'.format(data_file_path)):
        dictionary = joblib.load('{}/dictionary.txt'.format(data_file_path))
    else:
        print('No dictionary file found.')
        sys.exit(0)

    if os.path.exists('{}/lemmatizer.txt'.format(data_file_path)):
        lemmatizer = joblib.load('{}/lemmatizer.txt'.format(data_file_path))
    else:
        print('No lemmatizer file found.')
        sys.exit(0)

    for cluster_id, cluster_center in enumerate(kmodel.cluster_centers_):
        # Find the documents nearest to the cluster center.
        dist_vector = dist_space[:, cluster_id].argsort()
        nearest_doc_ids = dist_vector[:depth]

        tfidf_vectors = []
        for doc_id in nearest_doc_ids:
            tfidf_vector_dense = tfidf.getrow(doc_id).todense()
            tfidf_vectors.append(tfidf_vector_dense)
        # Find the terms in the cluster with the best cumulative Tf/Idf score.
        most_important_terms = [lemmatizer.stem2lemma(dictionary[term_id])
                                for term_id in top_terms(
                                    tfidf_vectors, depth, top_n=100)]
        cluster_reps.append(most_important_terms)

    # TODO: Refactor the filtering.
    changed = True
    while changed:
        changed = False
        # Make dict with term as key and frequency in all representations
        # as value.
        rep_terms = [term for rep in cluster_reps for term in rep[:3]]
        rep_term_frequencies = dict()
        rep_term_frequencies = Counter(rep_terms)

        # Filter out terms that appear in more than one cluster representation.
        # Stop filtering when the representation of a cluster has less than
        # 3 terms.
        filtered_cluster_reps = []
        removed_terms = []
        for index, rep in enumerate(cluster_reps):
            filtered_rep = []
            for term in rep:
                if (len(rep) - len(filtered_rep)) <= 3 or\
                        rep_term_frequencies[term] <= 1:
                    filtered_rep.append(term)
                else:
                    changed = True
                    removed_terms.append(term)
            filtered_cluster_reps.append(filtered_rep)

        cluster_reps = [rep[:3] for rep in filtered_cluster_reps]

    return cluster_reps


def get_cluster_category(kmodel, data_file_path, depth):
    """ Uses the categories assigned by Wikipedia to visualize the clusters.

    Finds the three most common categories in the closest documents to
    the cluster center.

    Args:
        kmodel (obj): A scikit-learn K-means model object.
        data_file_path (str): The path to the data directory.
        depth (int): The number of documents closest to the cluster center
            that will be processed.
    Returns:
        cluster_reps (list): A list that contains three category
            representations for each cluster.
        cluster_reps_percentages (list): A list of the percentages of occurence
            for the three most common categories in each cluster.

    """
    assert os.path.exists('{}/title_category.txt'.format(data_file_path))
    assert os.path.exists('{}/topic_space.txt'.format(data_file_path))
    assert os.path.exists('{}/title_dict.txt'.format(data_file_path))
    # Pandas dataframe that matches titles to categories.
    title_category_frame = joblib.load(
        '{}/title_category.txt'.format(data_file_path))
    vector_space = joblib.load('{}/topic_space.txt'.format(data_file_path))
    title_dict = joblib.load('{}/title_dict.txt'.format(data_file_path))
    dist_space = kmodel.transform(vector_space)

    cluster_reps = []
    cluster_reps_percentages = []
    for cluster_id, cluster_center in enumerate(kmodel.cluster_centers_):
        # Find the titles of the documents closest to the cluster center.
        dist_vector = dist_space[:, cluster_id].argsort()
        nearest_doc_ids = dist_vector[:depth]
        nearest_titles = [title_dict[doc_id] for doc_id in nearest_doc_ids]
        categories = []
        # Find the categories that occur in the documents.
        for title in nearest_titles:
            if title in title_category_frame['Title'].tolist():
                index = title_category_frame.index[
                    title_category_frame['Title'] == title].tolist()[0]
                categories.extend(title_category_frame.at[index, 'Category'])
        # Get frequencies of occurence for each category.
        counter = Counter(categories)
        cluster_reps_percentages.append(
            [round((category[1] / len(nearest_doc_ids)) * 100)
             for category in counter.most_common(3)])
        cluster_reps.append([category[0]
                             for category in counter.most_common(3)])
    return [cluster_reps, cluster_reps_percentages]
