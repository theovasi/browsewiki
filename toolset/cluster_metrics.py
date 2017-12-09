import argparse
import joblib
from sklearn import metrics


def cluster_metrics(kmodel, vector_space):
    """ Calculates unsupervised clustering metrics for a kmeans model.

    Args:
        kmodel (:obj: `sklearn.cluster.KMeans`): The kmeans model that
            will be evaluated.
        vector_space (sparse matrix): The vector space that the kmeans
            model was fit on.

    """
    silhouette = metrics.silhouette_score(vector_space, kmodel.labels_,
                                          metric='euclidean')
    calhar = metrics.calinski_harabaz_score(
        vector_space.toarray(), kmodel.labels_)

    return silhouette, calhar


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Parses data file path.')
    parser.add_argument('-k', '--kmodel', type=str,
                        help='Path to the kmeans model file.')
    parser.add_argument('-s', '--space', type=str,
                        help='Path to the vector space file.')
    parser.add_argument('-o', '--output', type=str,
                        help='Path to the directory where the output file\
                              will be stored.')
    args = parser.parse_args()
    kmodel = joblib.load(args.kmodel)
    vector_space = joblib.load(args.space)
    silhouette, calhar = cluster_metrics(kmodel, vector_space)

    with open('{}/metric_results.txt'.format(args.output), 'w+') as output:
        output.write('Silhouette coefficient: ' + str(silhouette))
        output.write('\nCaliski-Harabaz index: ' + str(calhar))
