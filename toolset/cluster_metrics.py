"Calculates Silhouette coefficient and Calinski-Harabaz index for a kmeans model."
import os, sys
import argparse, joblib
from sklearn import metrics

def cluster_metrics(data_file_path):
    if not os.path.exists(data_file_path + '/kmodel.txt'):
        print('No k-means model file found.')
        sys.exit(0)
    kmodel = joblib.load(data_file_path + '/kmodel.txt')

    # If no topic_space.txt file exists, the clustering was performed on the
    # Tf-Idf matrix so load that instead.
    if os.path.exists(data_file_path + '/topic_space.txt'):
        vector_space = joblib.load(data_file_path + '/topic_space.txt')
        print('Calculating metrics for topic vector space.')
    else:
        vector_space = joblib.load(data_file_path + '/tfidf_sparse.txt')
        print('Calculating metrics for Tf-Idf vector space.')

    silhouette = metrics.silhouette_score(vector_space, kmodel.labels_,
                                          metric='euclidean')
    calhar = metrics.calinski_harabaz_score(vector_space.toarray(), kmodel.labels_)

    with open(data_file_path + '/metric_results.txt', 'w+') as output:
        output.write('Silhouette coefficient: ' + str(silhouette))
        output.write('\nCaliski-Harabaz index: ' + str(calhar))



if __name__ == '__main__':
    parser = argparse.ArgumentParser('Parses data file path.')
    parser.add_argument('data_file_path', type=str,
                        help='The file to the data directory.')
    args = parser.parse_args()
    cluster_metrics(args.data_file_path)
