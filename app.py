import random
from flask import Flask, render_template, redirect, url_for, request, session
from flask_wtf import FlaskForm
from forms import ScatterGatherForm
import joblib, argparse
import numpy as np
import scipy.sparse as sp
from scipy.sparse import vstack
from sklearn.cluster import MiniBatchKMeans as mbk
from representations import get_cluster_reps


app = Flask(__name__)
app.secret_key = 'v3rys3cr3t'

@app.route('/')
@app.route('/index', methods=['GET', 'POST'])
def index():
    sgform = ScatterGatherForm()
    # Initialize cluster select/view form dynamically depending on the number of clusters.
    n_clusters = len(app.config['kmodel'].cluster_centers_)
    sgform.cluster_select.choices=[(i, 'cluster_{}'.format(i)) for i in range(n_clusters)]
    sgform.cluster_view.choices=[(i, 'cluster_{}'.format(i)) for i in range(n_clusters)]
    # Keep the ids of the documents used in this scatter iteration in a list.
    app.config['doc_ids'] = [i for i in range(app.config['vector_space'].shape[0])] 
    session['rand_test'] = random.random()

    if request.method == 'POST':
        app.logger.debug(session['rand_test'])
        if 'cluster_select' in request.form:
            selected_clusters = sgform.cluster_select.data 
            app.logger.debug(selected_clusters)
            # Get assignments of documents to clusters in a vector of cluster ids where
            # the document ids are the indices.
            labels = app.config['kmodel'].labels_

            # Gather the documents that are assigned to the selected clusters.
            doc_ids = []
            titles = []
            summaries = []
            links = []
            for i, label in enumerate(labels):
                if str(label) in selected_clusters:
                    doc_ids.append(i)
                    titles.append(app.config['titles'][i])
                    summaries.append(app.config['summaries'][i])
                    links.append(app.config['links'][i])
            app.config['doc_ids'] = doc_ids # This is the new scatter document collection.
            app.config['titles'] = titles
            app.config['summaries'] = summaries 
            app.config['links'] = links 

            # Create a new topic space matrix by selecting only the vector representations
            # of the new scatter collection documents.
            for doc_id in app.config['doc_ids']:
                doc_vector = app.config['vector_space'].getrow(doc_id)
                if 'scatter_vector_space' not in locals():
                    scatter_vector_space = sp.csr.csr_matrix(doc_vector)
                else:
                    scatter_vector_space = vstack([scatter_vector_space, doc_vector], format='csr')
            app.config['vector_space'] = scatter_vector_space

            # Perform the clustering using the new vector space.
            kmodel = mbk(n_clusters=app.config['k'], max_iter=10)
            kmodel.fit(scatter_vector_space)
            app.config['kmodel'] = kmodel
            app.config['dist_space'] = kmodel.transform(scatter_vector_space)

            # Get the representations of the clusters.
            for cluster_id in range(len(kmodel.cluster_centers_)):
                app.config['cluster_reps'] = get_cluster_reps(kmodel)

            return render_template('index.html', sgform=sgform, cluster_reps=app.config['cluster_reps'],
                                   select_list=list(sgform.cluster_select))

        elif 'cluster_view' in request.form:
            nearest_titles = []
            nearest_summaries = []
            nearest_links = []
            cluster_view_id = sgform.cluster_view.data[0]
            # Vector of document distances from the cluster center.
            dist_vector = app.config['dist_space'][:, int(cluster_view_id)]
            # The ids of the document nearest to the cluster center.
            nearest_doc_ids = dist_vector.argsort()[:50]
            for doc_id in nearest_doc_ids:
                nearest_titles.append(app.config['titles'][doc_id])
                nearest_summaries.append(app.config['summaries'][doc_id])
                nearest_links.append(app.config['links'][doc_id])

            return render_template('index.html', sgform=sgform, cluster_reps=app.config['cluster_reps'],
                                   select_list=list(sgform.cluster_select), titles=nearest_titles,
                                   summaries=nearest_summaries, links=nearest_links)
            

    app.config['cluster_reps'] = get_cluster_reps(app.config['kmodel'])
    return render_template('index.html', sgform=sgform, cluster_reps=app.config['cluster_reps'],
                           select_list=list(sgform.cluster_select))

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Process command line arguments.')
    parser.add_argument('data_file_path', type=str,
                        help='Path to the data file.')
    args = parser.parse_args()
    titles = []
    summaries = []
    links = []
    lda_model = joblib.load('{}/topic_model.txt'.format(args.data_file_path))
    dictionary = joblib.load('{}/dictionary.txt'.format(args.data_file_path))
    title_dict = joblib.load('{}/title_dict.txt'.format(args.data_file_path))
    link_dict = joblib.load('{}/link_dict.txt'.format(args.data_file_path))
    summary_dict = joblib.load('{}/summary_dict.txt'.format(args.data_file_path))
    for i in range(len(title_dict)):
        titles.append(title_dict[i])
        summaries.append(summary_dict[i])
        links.append(link_dict[i])
    app.config['titles'] = titles
    app.config['summaries'] = summaries
    app.config['links'] = links
    app.config['vector_space'] = joblib.load('{}/topic_space.txt'.format(args.data_file_path))
    app.config['kmodel'] = joblib.load('{}/kmodel.txt'.format(args.data_file_path))
    app.config['dist_space'] = joblib.load('{}/dist_space.txt'.format(args.data_file_path))
    # Constant that is used in the k(number of clusters) decision rule.
    app.config['k'] = len(app.config['kmodel'].cluster_centers_)
    app.config['cluster_reps'] = None
    app.run(debug=True)
