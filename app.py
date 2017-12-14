import uuid
from math import ceil, floor
from flask import Flask, render_template, redirect, url_for, request, session, send_from_directory
from flask_wtf import FlaskForm
from flask_kvsession import KVSessionExtension
from simplekv.fs import FilesystemStore
from forms import ScatterGatherForm, SearchForm
import joblib
import argparse
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse import vstack
from scipy.sparse import csc_matrix
from gensim import matutils
from sklearn.cluster import MiniBatchKMeans as mbk
from sklearn.neighbors import NearestNeighbors as nn
from toolset import visualize
from toolset.mogreltk import stem

store = FilesystemStore('.sessiondata')

app = Flask(__name__, static_url_path='/static')
app.secret_key = 'v3rys3cr3t'
KVSessionExtension(store, app)


class Pagination(object):
    """ Split a list of items into pages. """

    def __init__(self, source, cluster_view_id, current_page, items_per_page, n_items, query=None):
        self.source = source
        self.query = query
        self.cluster_view_id = cluster_view_id
        self.current_page = current_page
        self.items_per_page = items_per_page
        self.n_items = n_items

    def n_pages(self):
        return int(ceil(self.n_items / self.items_per_page))

    def has_prev(self):
        return self.current_page > 0

    def has_next(self):
        return self.current_page < self.n_pages()

    def iter_pages(self):
        if self.current_page < 6:
            for page_num in range(min(12, self.n_pages())):
                yield page_num
        elif self.current_page > (self.n_pages() - 6):
            for page_num in range(self.n_pages() - 12, self.n_pages()):
                yield page_num
        else:
            mid = min(6 + (self.current_page - 5), self.n_pages())
            for page_num in range(max(mid - 6, 0), min(mid + 6, self.n_pages())):
                yield page_num


def url_for_page(page):
    page_args = request.view_args.copy()
    page_args['current_page'] = page
    return url_for('view_page', **page_args)


app.jinja_env.globals['url_for_page'] = url_for_page


def docs_for_page(cluster_view_id, page):
    nearest_titles = []
    nearest_summaries = []
    nearest_links = []
    # Vector of document distances from the cluster center.
    dist_vector = session['dist_space'][:, int(cluster_view_id)]
    # The ids of the document nearest to the cluster center.
    nearest_doc_ids = dist_vector.argsort()
    filtered_nearest_doc_ids = []
    for doc_id in nearest_doc_ids:
        # Only keep documents that belong to the cluster in question.
        if session['kmodel'].labels_[doc_id] == int(cluster_view_id):
            filtered_nearest_doc_ids.append(doc_id)

    for doc_id in filtered_nearest_doc_ids[page * 16:(page + 1) * 16]:
        nearest_titles.append(session['titles'][doc_id])
        nearest_summaries.append(session['summaries'][doc_id])
        nearest_links.append(session['links'][doc_id])

    return len(filtered_nearest_doc_ids), nearest_titles, nearest_summaries, nearest_links


def k_nearest_docs_for_page(query, cluster_view_id, page):
    nearest_titles = []
    nearest_summaries = []
    nearest_links = []
    vector = session['tfidf_model'][session['dictionary'].doc2bow(
        stem(query))]
    vector.append((session['tfidf'].shape[1] - 1, 0))
    vector_sparse = csc_matrix.transpose(
        matutils.corpus2csc([vector])).tocsr()
    nearest_doc_ids = session['nn_model'].kneighbors(
        vector_sparse)[1][0]
    # The ids of the document nearest to the cluster center.
    filtered_nearest_doc_ids = []
    for doc_id in nearest_doc_ids:
        # Only keep documents that belong to the cluster in question.
        if session['kmodel'].labels_[doc_id] == int(cluster_view_id):
            filtered_nearest_doc_ids.append(doc_id)

    for doc_id in filtered_nearest_doc_ids[page * 16:(page + 1) * 16]:
        nearest_titles.append(session['titles'][doc_id])
        nearest_summaries.append(session['summaries'][doc_id])
        nearest_links.append(session['links'][doc_id])

    return len(filtered_nearest_doc_ids), nearest_titles, nearest_summaries, nearest_links


@app.route('/')
def session_init():
    session.clear()
    session['uid'] = uuid.uuid4()
    session['tfidf_model'] = app.config['tfidf_model']
    session['tfidf'] = app.config['tfidf']
    session['dictionary'] = app.config['dictionary']
    session['nn_model'] = app.config['nn_model']
    session['vector_space'] = app.config['vector_space']
    session['kmodel'] = app.config['kmodel']
    session['dist_space'] = app.config['dist_space']
    session['doc_ids'] = app.config['doc_ids']
    session['titles'] = app.config['titles']
    session['summaries'] = app.config['summaries']
    session['links'] = app.config['links']
    session['k'] = app.config['k']
    return redirect(url_for('index'))


@app.route('/index', methods=['GET', 'POST'])
def index(current_page=0):
    app.logger.debug(session['uid'])
    app.logger.debug(current_page)
    sgform = ScatterGatherForm()
    search_form = SearchForm()

    # Initialize cluster select/view form dynamically depending on the number
    # of clusters.
    n_clusters = len(app.config['kmodel'].cluster_centers_)
    sgform = ScatterGatherForm()
    sgform.cluster_select.choices = [(i, 'cluster_{}'.format(i))
                                     for i in range(n_clusters)]
    sgform.cluster_view.choices = [(i, 'cluster_{}'.format(i))
                                   for i in range(n_clusters)]

    if request.method == 'POST':
        if 'query' in request.form:
            query = request.form['query']
            app.logger.debug('Query received: ' + str(request.form['query']))
            n_docs, nearest_titles, nearest_summaries, nearest_links =\
                k_nearest_docs_for_page(
                    query, session['pagination'].cluster_view_id, current_page)
            session['pagination'] =\
                Pagination('search', session['pagination'].cluster_view_id,
                           current_page, 16, n_docs, query=query)
            sgform = ScatterGatherForm()
            sgform.cluster_select.choices = [(i, 'cluster_{}'.format(i))
                                             for i in range(n_clusters)]
            sgform.cluster_view.choices = [(i, 'cluster_{}'.format(i))
                                           for i in range(n_clusters)]
            return render_template('index.html', sgform=sgform,
                                   search_form=search_form,
                                   cluster_reps=session['cluster_reps'],
                                   select_list=list(sgform.cluster_select),
                                   n_docs=n_docs,
                                   pagination=session['pagination'],
                                   titles=nearest_titles,
                                   summaries=nearest_summaries,
                                   links=nearest_links,
                                   cluster_doc_counts=session[
                                       'cluster_doc_counts'])

        elif 'cluster_select' in request.form:
            selected_clusters = sgform.cluster_select.data
            app.logger.debug('Selected: ' + str(selected_clusters))
            # Get assignments of documents to clusters in a vector of cluster
            # ids where the document ids are the indices.
            labels = session['kmodel'].labels_

            # Gather the documents that are assigned to the selected clusters.
            doc_ids = []
            titles = []
            summaries = []
            links = []
            app.logger.debug('Gathering document representations...')
            for i, label in enumerate(labels):
                if str(label) in selected_clusters:
                    doc_ids.append(i)
                    titles.append(session['titles'][i])
                    summaries.append(session['summaries'][i])
                    links.append(session['links'][i])
            if len(doc_ids) < (session['k'] * 16):

                sgform = ScatterGatherForm()
                sgform.cluster_select.choices = [(i, 'cluster_{}'.format(i))
                                                 for i in range(session['k'])]
                sgform.cluster_view.choices = [(i, 'cluster_{}'.format(i))
                                               for i in range(session['k'])]
                return render_template('index.html', sgform=sgform,
                                       search_form=search_form,
                                       cluster_reps=session['cluster_reps'],
                                       select_list=list(sgform.cluster_select),
                                       cluster_doc_counts=session[
                                           'cluster_doc_counts'])

            # This is the new scatter document collection.
            session['doc_ids'] = doc_ids
            session['titles'] = titles
            session['summaries'] = summaries
            session['links'] = links

            app.logger.debug('Constructing new vector space...')
            # Create a new topic space matrix by selecting only the vector
            # representations of the new scatter collection documents.
            for doc_id in session['doc_ids']:
                doc_vector = session['vector_space'].getrow(doc_id)
                tfidf_vector = session['tfidf'].getrow(doc_id)
                if 'scatter_vector_space' not in locals():
                    scatter_vector_space = sp.csr.csr_matrix(doc_vector)
                    scatter_tfidf = sp.csr.csr_matrix(tfidf_vector)
                else:
                    scatter_vector_space =\
                        vstack([scatter_vector_space, doc_vector],
                               format='csr')
                    scatter_tfidf =\
                        vstack([scatter_tfidf, tfidf_vector], format='csr')

            session['tfidf'] = scatter_tfidf
            session['nn_model'].fit(scatter_tfidf)
            session['vector_space'] = scatter_vector_space

            app.logger.debug('Clustering...')
            # Perform the clustering using the new vector space.
            total_corpus_size = len(app.config['doc_ids'])
            gathered_corpus_size = len(session['doc_ids'])
            coeff = 3
            session['k'] = round(
                floor((total_corpus_size +
                       gathered_corpus_size) / (coeff * pow(10, 4))))
            kmodel = mbk(n_clusters=session['k'], max_iter=10)
            kmodel.fit(scatter_vector_space)
            session['kmodel'] = kmodel
            session['dist_space'] = kmodel.transform(scatter_vector_space)

            # Count number of documents in each cluster.
            cluster_doc_counts = [0 for i in range(
                len(session['kmodel'].cluster_centers_))]

            for label in session['kmodel'].labels_:
                cluster_doc_counts[label] += 1
            session['cluster_doc_counts'] = cluster_doc_counts

            app.logger.debug('Generating cluster representations...')
            # Get the representations of the clusters.
            session['cluster_reps'] =\
                visualize.get_cluster_reps(session['tfidf'],
                                           session['kmodel'],
                                           session['dist_space'],
                                           app.config['data_file_path'], 50)

            sgform = ScatterGatherForm()
            sgform.cluster_select.choices = [(i, 'cluster_{}'.format(i))
                                             for i in range(session['k'])]
            sgform.cluster_view.choices = [(i, 'cluster_{}'.format(i))
                                           for i in range(session['k'])]

            return render_template('index.html', sgform=sgform,
                                   search_form=search_form,
                                   cluster_reps=session['cluster_reps'],
                                   select_list=list(sgform.cluster_select),
                                   cluster_doc_counts=session[
                                       'cluster_doc_counts'])

        elif 'cluster_view' in request.form:
            cluster_view_id = sgform.cluster_view.data[0]
            app.logger.debug(cluster_view_id)
            n_docs, nearest_titles, nearest_summaries, nearest_links =\
                docs_for_page(cluster_view_id, current_page)
            session['pagination'] = Pagination('view',
                                               cluster_view_id, current_page,
                                               16, n_docs)
            sgform = ScatterGatherForm()
            sgform.cluster_select.choices = [(i, 'cluster_{}'.format(i))
                                             for i in range(session['k'])]
            sgform.cluster_view.choices = [(i, 'cluster_{}'.format(i))
                                           for i in range(session['k'])]
            return render_template('index.html', sgform=sgform,
                                   search_form=search_form,
                                   cluster_reps=session['cluster_reps'],
                                   select_list=list(sgform.cluster_select),
                                   n_docs=n_docs,
                                   pagination=session['pagination'],
                                   titles=nearest_titles,
                                   summaries=nearest_summaries,
                                   links=nearest_links,
                                   cluster_doc_counts=session[
                                       'cluster_doc_counts'])

    session['cluster_reps'] =\
        visualize.get_cluster_reps(session['tfidf'],
                                   session['kmodel'],
                                   session['dist_space'],
                                   app.config['data_file_path'],
                                   50)

    # Count number of documents in each cluster.
    cluster_doc_counts = [0 for i in range(
        len(session['kmodel'].cluster_centers_))]
    for label in session['kmodel'].labels_:
        cluster_doc_counts[label] += 1
    session['cluster_doc_counts'] = cluster_doc_counts

    return render_template('index.html', sgform=sgform,
                           search_form=search_form,
                           cluster_reps=session['cluster_reps'],
                           select_list=list(sgform.cluster_select),
                           cluster_doc_counts=session['cluster_doc_counts'])


@app.route('/view_page/<int:current_page>', methods=['GET', 'POST'])
def view_page(current_page):
    sgform = ScatterGatherForm()
    search_form = SearchForm()

    n_clusters = len(app.config['kmodel'].cluster_centers_)
    sgform.cluster_select.choices = [(i, 'cluster_{}'.format(i))
                                     for i in range(n_clusters)]
    sgform.cluster_view.choices = [(i, 'cluster_{}'.format(i))
                                   for i in range(n_clusters)]

    app.logger.debug(session['pagination'].source)
    if session['pagination'].source == 'search':
        n_docs, nearest_titles, nearest_summaries, nearest_links =\
            k_nearest_docs_for_page(session['pagination'].query,
                                    session['pagination'].cluster_view_id,
                                    current_page)
    else:
        n_docs, nearest_titles, nearest_summaries, nearest_links =\
            docs_for_page(
                session['pagination'].cluster_view_id, current_page)
    session['pagination'].current_page = current_page
    return render_template('index.html', sgform=sgform,
                           search_form=search_form,
                           cluster_reps=session['cluster_reps'],
                           select_list=list(sgform.cluster_select),
                           n_docs=session['pagination'].n_items,
                           pagination=session['pagination'],
                           titles=nearest_titles,
                           summaries=nearest_summaries,
                           links=nearest_links,
                           cluster_doc_counts=session['cluster_doc_counts'])


@app.route('/view_page/static/<path:path>')
def send_html(path):
    return send_from_directory('static', path)


if __name__ == '__main__':
    data_file_path = './appdata'
    doc_ids = []
    titles = []
    summaries = []
    links = []
    title_dict = joblib.load('{}/title_dict.txt'.format(data_file_path))
    link_dict = joblib.load('{}/link_dict.txt'.format(data_file_path))
    summary_dict = joblib.load(
        '{}/summary_dict.txt'.format(data_file_path))
    for i in range(len(title_dict)):
        doc_ids.append(i)
        titles.append(title_dict[i])
        summaries.append(summary_dict[i])
        links.append(link_dict[i])
    app.config['data_file_path'] = data_file_path
    app.config['doc_ids'] = doc_ids
    app.config['titles'] = titles
    app.config['summaries'] = summaries
    app.config['links'] = links
    app.config['tfidf_model'] = joblib.load(
        '{}/tfidf_model.txt'.format(data_file_path))
    app.config['tfidf'] = joblib.load(
        '{}/tfidf_sparse.txt'.format(data_file_path))
    # Train nearest neighbors model.
    app.config['nn_model'] = nn(n_neighbors=1000, radius=10)
    app.config['nn_model'].fit(app.config['tfidf'])

    app.config['dictionary'] = joblib.load(
        '{}/dictionary.txt'.format(data_file_path))
    app.config['vector_space'] = joblib.load(
        '{}/topic_space.txt'.format(data_file_path))
    app.config['kmodel'] = joblib.load(
        '{}/kmodel.txt'.format(data_file_path))
    app.config['dist_space'] = joblib.load(
        '{}/dist_space.txt'.format(data_file_path))
    # Constant that is used in the k(number of clusters) decision rule.
    app.config['k'] = len(app.config['kmodel'].cluster_centers_)
    app.config['cluster_reps'] = None
    app.run(debug=True)
