import os
import sys
import joblib
import numpy as np
import random

from toolset import mogreltk
from collections import Counter
from operator import add


def top_terms(tfidf_vectors, top_n=3):
    """ Find terms with the highest cumulative Tf-Idf score across all vectors.

        Args:
            document_vectors (list): A list of document Tf-Idf representations.
            top_n (int): The number of highest scoring terms to be returned.

        Returns:
            top_terms (list): The terms with the highest cumulative Tf-Idf score.
    """
    tfidf_sum = []
    for index, vector in enumerate(tfidf_vectors):
        if index == 0:
            tfidf_sum = np.array(vector)
        else:
            tfidf_sum = tfidf_sum + np.array(vector)

    return np.argsort(tfidf_sum).tolist()[::-1][:top_n]


def get_cluster_reps(tfidf, kmodel, topic_space, dictionary, lemmatizer, depth=100):
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
    for cluster_id, cluster_center in enumerate(kmodel.cluster_centers_):
        # Find the documents nearest to the cluster center.
        tfidf_vectors = []

        cluster_doc_ids = []
        dist_space = kmodel.transform(topic_space)
        dist_vector = dist_space[:, int(cluster_id)].argsort()
        nearest_doc_ids = []
        for doc_id in dist_vector:
            if kmodel.labels_[doc_id] == cluster_id:
                nearest_doc_ids.append(doc_id)
        cluster_doc_ids.extend(nearest_doc_ids[:50])

        for doc_id, label in enumerate(kmodel.labels_):
            if label == cluster_id:
                cluster_doc_ids.append(doc_id)
        # 5% of documents in cluster.
        sample_size = int(len(cluster_doc_ids) * 0.2)
        random_sample_ids = random.sample(cluster_doc_ids, sample_size)

        for doc_id in random_sample_ids:
            tfidf_vector_dense = tfidf.getrow(doc_id).todense().tolist()[0]
            tfidf_vectors.append(tfidf_vector_dense)
        # Find the terms in the cluster with the best cumulative Tf/Idf score.
        most_important_terms = [lemmatizer.stem2lemma(dictionary[term_id])
                                for term_id in top_terms(
                                    tfidf_vectors, top_n=200)]
        cluster_reps.append(most_important_terms)


    changed = True
    removed_terms = []
    while changed:
        changed = False
        rep_terms = [term for rep in cluster_reps for term in rep[:3]]
        rep_term_frequencies = Counter(rep_terms)

        # Filter out terms that appear in more than one cluster representation.
        filtered_cluster_reps = []

        for index, rep in enumerate(cluster_reps):
            filtered_rep = []

            if len(rep) <= 3:
                filtered_cluster_reps.append(rep)
                continue

            for term in rep:
                if rep_term_frequencies[term] < 2:
                    filtered_rep.append(term)
                else:
                    removed_terms.append(term)
                    changed = True
            filtered_cluster_reps.append(filtered_rep)

        cluster_reps = [rep for rep in filtered_cluster_reps]

    cluster_reps = [rep[:3] for rep in cluster_reps]
    removed_term_freq = Counter(removed_terms)
    most_common_removed = [term_tuple for term_tuple in removed_term_freq]
    return cluster_reps, most_common_removed[:3]


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
