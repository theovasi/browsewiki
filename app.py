from flask import Flask, render_template, redirect, url_for, request, session
from flask_wtf import FlaskForm
from forms import ScatterGatherForm
import joblib, argparse
import numpy as np
import scipy.sparse as sp
from sklearn.cluster import MiniBatchKMeans as mbk


app = Flask(__name__)
app.secret_key = 'v3rys3cr3t'

@app.route('/')
@app.route('/index', methods=['GET', 'POST'])
def index():
    cluster_reps = dict()
    absolute_doc_id_dict = dict()
    tfidf_sparse = joblib.load('{}/topic_space.txt'.format(args.data_file_path))
    kmodel = joblib.load('{}/kmodel.txt'.format(args.data_file_path))
    dist_space = joblib.load('{}/dist_space.txt'.format(args.data_file_path))
    sgform = ScatterGatherForm()
    n_clusters = len(kmodel.cluster_centers_)
    sgform.cluster_select.choices=[(i, 'cluster_{}'.format(i)) for i in range(n_clusters)]
    sgform.cluster_view.choices=[(i, 'cluster_{}'.format(i)) for i in range(n_clusters)]

    for ind in range(tfidf_sparse.shape[0]):
        absolute_doc_id_dict[ind] = ind
    if request.method == 'POST':
        doc_ids = []
        titles = []
        links = []
        summaries = []
        if 'cluster_select' in request.form:
            selected_clusters = sgform.cluster_select.data
            app.logger.debug('\n\n\nSelected: ' + ' '.join(selected_clusters))
            new_tfidf = []
            abs_ind = 0
            tmp_dict = dict()
            for selected_cluster_id in selected_clusters:
                for doc_id, cluster_id in enumerate(kmodel.labels_):
                    if cluster_id == int(selected_cluster_id):
                        tmp_dict[abs_ind] = absolute_doc_id_dict[doc_id]
                        new_tfidf = sp.vstack((new_tfidf, tfidf_sparse[doc_id]),
                                              format='csr')
            tfidf_sparse = new_tfidf[1:]
            absolute_doc_id_dict = dict(tmp_dict)
            kmodel = mbk(n_clusters=len(kmodel.cluster_centers_), max_iter=10,
                         verbose=True)
            kmodel.fit(tfidf_sparse)
            dist_space = kmodel.transform(tfidf_sparse)
            print(dist_space.shape)
            for i in range(len(kmodel.cluster_centers_)):
                terms = []
                topic_id = kmodel.cluster_centers_[i].argsort()[::-1][0]
                term_ids = lda_model.get_topic_terms(topic_id, topn=3)
                for term_id in term_ids:
                    terms.append(dictionary[term_id[0]])
                cluster_reps[i] = terms

            return render_template('index.html', sgform=sgform, titles=titles,
                               links=links, summaries=summaries, cluster_reps=cluster_reps,
                               select_list=list(sgform.cluster_select))

        if 'cluster_view' in request.form:
            cluster_id = request.form['cluster_view']
            app.logger.debug('\n\n\nView: ' + cluster_id)
            dist = dist_space[:, int(cluster_id)] # Distances of the documents form the cluster center.
            # Find the ids of the documents that belogn to this cluster.
            doc_ids = dist.argsort()[:50]
            for id in doc_ids:
                titles.append(title_dict[absolute_doc_id_dict[id]])
                links.append(link_dict[absolute_doc_id_dict[id]])
                summaries.append(summary_dict[absolute_doc_id_dict[id]])
            sgform.cluster_select.data = None
            for i in range(len(kmodel.cluster_centers_)):
                terms = []
                topic_id = kmodel.cluster_centers_[i].argsort()[::-1][0]
                term_ids = lda_model.get_topic_terms(topic_id, topn=3)
                for term_id in term_ids:
                    terms.append(dictionary[term_id[0]])
                cluster_reps[i] = terms

            return render_template('index.html', sgform=sgform, titles=titles,
                               links=links, summaries=summaries, cluster_reps=cluster_reps,
                               select_list=list(sgform.cluster_select))

    for i in range(len(kmodel.cluster_centers_)):
        terms = []
        topic_id = kmodel.cluster_centers_[i].argsort()[::-1][0]
        term_ids = lda_model.get_topic_terms(topic_id, topn=3)
        for term_id in term_ids:
            terms.append(dictionary[term_id[0]])
        cluster_reps[i] = terms
    return render_template('index.html', sgform=sgform, cluster_reps=cluster_reps,
                           select_list=list(sgform.cluster_select))

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Process command line arguments.')
    parser.add_argument('data_file_path', type=str,
                        help='Path to the data file.')
    args = parser.parse_args()
    lda_model = joblib.load('{}/lda_model.txt'.format(args.data_file_path))
    dictionary = joblib.load('{}/dictionary.txt'.format(args.data_file_path))
    title_dict = joblib.load('{}/title_dict.txt'.format(args.data_file_path))
    link_dict = joblib.load('{}/link_dict.txt'.format(args.data_file_path))
    summary_dict = joblib.load('{}/summary_dict.txt'.format(args.data_file_path))
    app.run(debug=True)
