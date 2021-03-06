import os
import uuid
from math import ceil, floor
from flask import Flask, render_template, redirect, url_for, request, session,\
    send_from_directory
from flask_kvsession import KVSessionExtension
from simplekv.fs import FilesystemStore
from forms import ScatterGatherForm, SearchForm
import joblib
from scipy.sparse import vstack
from scipy.sparse import csc_matrix
from gensim import matutils
from sklearn.cluster import MiniBatchKMeans as mbk
from sklearn.neighbors import NearestNeighbors as nn
from sklearn.metrics.pairwise import euclidean_distances as edist
from toolset import visualize
from toolset import mogreltk


store = FilesystemStore('.sessiondata')

app = Flask(__name__, static_url_path='/static')
app.secret_key = os.urandom(24)
KVSessionExtension(store, app)


class Pagination(object):
    """ Split a list of items into pages. """

    def __init__(self, source, result, cluster_view_id,
                 current_page, items_per_page, n_items, query=None):
        self.source = source
        self.result = result
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
        if self.current_page < 4:
            for page_num in range(min(6, self.n_pages())):
                yield page_num
        elif self.current_page > (self.n_pages() - 4):
            for page_num in range(self.n_pages() - 6, self.n_pages()):
                yield page_num
        else:
            mid = min(4 + (self.current_page - 3), self.n_pages())
            for page_num in range(max(mid - 4, 0), min(mid + 2, self.n_pages())):
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

    for doc_id in filtered_nearest_doc_ids[page * 14:(page + 1) * 14]:
        nearest_titles.append(session['titles'][doc_id])
        nearest_summaries.append(session['summaries'][doc_id])
        nearest_links.append(session['links'][doc_id])

    return len(filtered_nearest_doc_ids), nearest_titles, nearest_summaries, nearest_links


def k_nearest_docs_for_page(query, cluster_view_id, page):
    nearest_titles = []
    nearest_summaries = []
    nearest_links = []
    tfidf_model = joblib.load(
        '{}/tfidf_model.txt'.format(data_file_path))
    dictionary = joblib.load('{}/dictionary.txt'.format(data_file_path))
    vector = tfidf_model[dictionary.doc2bow(mogreltk.stem(query))]
    vector.append((session['tfidf'].shape[1] - 1, 0))
    vector_sparse = csc_matrix.transpose(
        matutils.corpus2csc([vector])).tocsr()

    session['nn_model'] = nn(n_neighbors=min(1000, session['tfidf'].shape[0]))
    session['nn_model'].fit(session['tfidf'])
    nearest_doc_ids = session['nn_model'].kneighbors(
        vector_sparse)[1][0]
    # The ids of the document nearest to the cluster center.
    filtered_nearest_doc_ids = []
    for doc_id in nearest_doc_ids:
        # Only keep documents that belong to the cluster in question.
        if session['kmodel'].labels_[doc_id] == int(cluster_view_id):
            filtered_nearest_doc_ids.append(doc_id)

    for doc_id in filtered_nearest_doc_ids[page * 14:(page + 1) * 14]:
        nearest_titles.append(session['titles'][doc_id])
        nearest_summaries.append(session['summaries'][doc_id])
        nearest_links.append(session['links'][doc_id])

    return len(filtered_nearest_doc_ids), nearest_titles, nearest_summaries, nearest_links


def gather(session, selected_clusters):
    """ Update corpus data keeping only documents from selected clusters."""
    doc_ids = []
    titles = []
    summaries = []
    links = []
    doc_vector_list = []
    tfidf_vector_list = []
    for i, label in enumerate(session['kmodel'].labels_):
        if str(label) in selected_clusters:
            doc_ids.append(i)
            titles.append(session['titles'][i])
            summaries.append(session['summaries'][i])
            links.append(session['links'][i])

            # Create a new topic space matrix by selecting only the vector
            # representations of the new scatter collection documents.
            doc_vector_list.append(session['vector_space'].getrow(i))
            tfidf_vector_list.append(session['tfidf'].getrow(i))

    vector_space = vstack(doc_vector_list, format='csr')
    tfidf = vstack(tfidf_vector_list, format='csr')

    return doc_ids, titles, summaries, links, vector_space, tfidf


def kmeans_rule(total_corpus_size, gathered_corpus_size):
    coeff = 2
    return round(
        floor((total_corpus_size +
               gathered_corpus_size) / (coeff * pow(10, 4))))


def cluster_optimal_k(vector_space):
    """ Applies K-means clustering with an optimal K value.

    The optimal K is determined by applying K-means clustering and
    decreasing the value of K until the distance between all the
    cluster centers is bigger than a threshold.

    Args:
        vector_space(sparse matrix): The vector representation of the text
            collection.

    Returns:
        kmodel (:obj: `sklearn.cluster.MiniBatchKMeans`): A K-means model.

    """

    # Number of clusters must smaller than the number of documents.
    if vector_space.shape[0] < 28:
        n_clusters = 1
    else:
        n_clusters = 12

    kmodel = mbk(n_clusters=n_clusters, max_iter=100)
    kmodel.fit(vector_space)

    while check_cluster_distance(kmodel) and n_clusters > 1:
        n_clusters -= 1
        kmodel = mbk(n_clusters=n_clusters, max_iter=100)
        kmodel.fit(vector_space)

    return kmodel


def check_cluster_distance(kmodel, threshold=0.12):
    """ Check if two cluster centers are too close.

    Returns True when the distance between at least two two cluster
    centers in the K-means model is smaller than a threshold.


    Args:
        kmodel (:obj: `sklearn.cluster.MiniBatchKMeans`): A K-means model.

    """
    edist_space = edist(kmodel.cluster_centers_)
    for dist_vector in edist_space:
        for distance in list(dist_vector):
            if distance > 0 and distance < threshold:
                return True
    return False


@app.route('/')
def session_init():
    session.clear()
    session['uid'] = uuid.uuid4()
    session['tfidf'] = app.config['tfidf']
    session['nn_model'] = app.config['nn_model']
    session['vector_space'] = app.config['vector_space']
    session['kmodel'] = app.config['kmodel']
    session['cluster_reps'] = app.config['cluster_reps']
    session['common_terms'] = []
    session['doc_ids'] = app.config['doc_ids']
    session['titles'] = app.config['titles']
    session['summaries'] = app.config['summaries']
    session['links'] = app.config['links']
    session['k'] = app.config['k']
    return redirect(url_for('index'))


@app.route('/index', methods=['GET', 'POST'])
def index(current_page=0):
    sgform = ScatterGatherForm()
    search_form = SearchForm()

    # Initialize cluster select/view form dynamically depending on the number
    # of clusters.
    sgform = ScatterGatherForm()
    sgform.cluster_select.choices = [(i, 'cluster_{}'.format(i))
                                     for i in range(session['k'])]
    sgform.cluster_view.choices = [(i, 'cluster_{}'.format(i))
                                   for i in range(session['k'])]

    if request.method == 'POST':
        if 'query' in request.form:
            query = request.form['query']
            app.logger.debug('Query received: ' + str(request.form['query']))
            n_docs, nearest_titles, nearest_summaries, nearest_links =\
                k_nearest_docs_for_page(
                    query, session['pagination'].cluster_view_id, current_page)

            if session['k'] == 1:
                result = True
                template = 'result.html'
            else:
                result = False
                template = 'index.html'

            session['pagination'] =\
                Pagination('search', result, session['pagination'].cluster_view_id,
                           current_page, 16, n_docs, query=query)

            sgform = ScatterGatherForm()
            sgform.cluster_select.choices = [(i, 'cluster_{}'.format(i))
                                             for i in range(session['k'])]
            sgform.cluster_view.choices = [(i, 'cluster_{}'.format(i))
                                           for i in range(session['k'])]

            if result:
                return render_template(template, sgform=sgform,
                                       search_form=search_form,
                                       cluster_reps=session['cluster_reps'],
                                       view_list=list(sgform.cluster_view),
                                       rep=session['rep'],
                                       doc_count=session['rep_doc_count'],
                                       common_terms=session['common_terms'],
                                       n_docs=n_docs,
                                       pagination=session['pagination'],
                                       titles=nearest_titles,
                                       summaries=nearest_summaries,
                                       links=nearest_links,
                                       cluster_doc_counts=session[
                                           'cluster_doc_counts'])
            else:
                return render_template(template, sgform=sgform,
                                       search_form=search_form,
                                       cluster_reps=session['cluster_reps'],
                                       view_list=list(sgform.cluster_view),
                                       n_docs=n_docs,
                                       pagination=session['pagination'],
                                       common_terms=session['common_terms'],
                                       titles=nearest_titles,
                                       summaries=nearest_summaries,
                                       links=nearest_links,
                                       cluster_doc_counts=session[
                                           'cluster_doc_counts'])

        elif 'cluster_select' in request.form:
            selected_clusters = sgform.cluster_select.data
            app.logger.debug('Selected: ' + str(selected_clusters))

            session['doc_ids'], session['titles'], session['summaries'],\
                session['links'], session['vector_space'], session['tfidf'] =\
                gather(session, selected_clusters)
            session['nn_model'].fit(session['tfidf'])

            # Perform the clustering using the new vector space.
            session['kmodel'] = cluster_optimal_k(session['vector_space'])
            session['k'] = len(session['kmodel'].cluster_centers_)
            session['dist_space'] = session['kmodel'].transform(
                session['vector_space'])

            if session['k'] == 1:
                app.logger.debug('\nReached end\n')
                session['rep'] = session['cluster_reps'][int(
                    selected_clusters[0])]
                session['rep_doc_count'] = session['cluster_doc_counts'][
                    int(selected_clusters[0])]
                n_docs, nearest_titles, nearest_summaries, nearest_links =\
                    docs_for_page(0, current_page)
                session['pagination'] = Pagination('view', True,
                                                   0, current_page,
                                                   14, n_docs)
                return render_template('result.html',
                                       search_form=search_form,
                                       rep=session['rep'],
                                       doc_count=session['rep_doc_count'],
                                       common_terms=session['common_terms'],
                                       n_docs=n_docs,
                                       pagination=session['pagination'],
                                       titles=nearest_titles,
                                       summaries=nearest_summaries,
                                       links=nearest_links,
                                       )

            # Count number of documents in each cluster.
            session['cluster_doc_counts'] = []
            for i in range(len(session['kmodel'].cluster_centers_)):
                session['cluster_doc_counts'].append(
                    list(session['kmodel'].labels_).count(i))

            # Get representations for the clusters.
            session['cluster_reps'], session['common_terms'] = \
                visualize.get_cluster_reps(session['tfidf'],
                                           session['kmodel'],
                                           session['vector_space'],
                                           joblib.load('{}/dictionary.txt'.format(
                                               app.config['data_file_path'])),
                                           joblib.load('{}/lemmatizer.txt'.format(
                                               app.config['data_file_path'])), 100)

            # Initialize new sgform for new k.
            sgform = ScatterGatherForm()
            sgform.cluster_select.data = []
            sgform.cluster_view.data = []
            sgform.cluster_select.choices = [(i, 'cluster_{}'.format(i))
                                             for i in range(session['k'])]
            sgform.cluster_view.choices = [(i, 'cluster_{}'.format(i))
                                           for i in range(session['k'])]

            return render_template('index.html', sgform=sgform,
                                   search_form=search_form,
                                   cluster_reps=session['cluster_reps'],
                                   common_terms=session['common_terms'],
                                   view_list=list(sgform.cluster_view),
                                   cluster_doc_counts=session['cluster_doc_counts'])

        elif 'cluster_view' in request.form:
            cluster_view_id = sgform.cluster_view.data
            app.logger.debug(cluster_view_id)
            n_docs, nearest_titles, nearest_summaries, nearest_links = docs_for_page(
                cluster_view_id, current_page)
            session['pagination'] = Pagination('view', False,
                                               cluster_view_id, current_page,
                                               14, n_docs)

            # Clear sgform selected clusters.
            sgform = ScatterGatherForm()
            sgform.cluster_select.data = []
            sgform.cluster_select.choices = [(i, 'cluster_{}'.format(i))
                                             for i in range(session['k'])]
            sgform.cluster_view.choices = [(i, 'cluster_{}'.format(i))
                                           for i in range(session['k'])]
            return render_template('index.html', sgform=sgform,
                                   search_form=search_form,
                                   cluster_reps=session['cluster_reps'],
                                   common_terms=session['common_terms'],
                                   view_list=list(sgform.cluster_view),
                                   n_docs=n_docs,
                                   pagination=session['pagination'],
                                   titles=nearest_titles,
                                   summaries=nearest_summaries,
                                   links=nearest_links,
                                   cluster_doc_counts=session[
                                       'cluster_doc_counts'])

    session['dist_space'] = session['kmodel'].transform(
        session['vector_space'])

    # Count number of documents in each cluster.
    session['cluster_doc_counts'] = []
    for i in range(len(session['kmodel'].cluster_centers_)):
        session['cluster_doc_counts'].append(
            list(session['kmodel'].labels_).count(i))

    return render_template('index.html', sgform=sgform,
                           search_form=search_form,
                           cluster_reps=session['cluster_reps'],
                           view_list=list(sgform.cluster_view),
                           cluster_doc_counts=session['cluster_doc_counts'])


@app.route('/view_page/<int:current_page>', methods=['GET', 'POST'])
def view_page(current_page):
    sgform = ScatterGatherForm()
    search_form = SearchForm()

    sgform.cluster_select.choices = [(i, 'cluster_{}'.format(i))
                                     for i in range(session['k'])]
    sgform.cluster_view.choices = [(i, 'cluster_{}'.format(i))
                                   for i in range(session['k'])]

    app.logger.debug(session['pagination'].source)
    if session['pagination'].source == 'search':
        n_docs, nearest_titles, nearest_summaries, nearest_links = \
            k_nearest_docs_for_page(session['pagination'].query,
                                    session['pagination'].cluster_view_id,
                                    current_page)
    else:
        n_docs, nearest_titles, nearest_summaries, nearest_links = docs_for_page(
            session['pagination'].cluster_view_id, current_page)
    session['pagination'].current_page = current_page

    if session['pagination'].result:
        template = 'result.html'
        return render_template(template, sgform=sgform,
                               search_form=search_form,
                               cluster_reps=session['cluster_reps'],
                               view_list=list(sgform.cluster_view),
                               rep=session['rep'],
                               doc_count=session['rep_doc_count'],
                               common_terms=session['common_terms'],
                               n_docs=session['pagination'].n_items,
                               pagination=session['pagination'],
                               titles=nearest_titles,
                               summaries=nearest_summaries,
                               links=nearest_links,
                               cluster_doc_counts=session['cluster_doc_counts'])
    else:
        template = 'index.html'
        return render_template(template, sgform=sgform,
                               search_form=search_form,
                               cluster_reps=session['cluster_reps'],
                               view_list=list(sgform.cluster_view),
                               n_docs=session['pagination'].n_items,
                               pagination=session['pagination'],
                               common_terms=session['common_terms'],
                               titles=nearest_titles,
                               summaries=nearest_summaries,
                               links=nearest_links,
                               cluster_doc_counts=session['cluster_doc_counts'])


@app.route('/view_page/static/<path:path>')
def send_html(path):
    return send_from_directory('static', path)


if __name__ == '__main__':
    data_file_path = './appdata'
    corpus_frame = joblib.load('{}/corpus_frame.txt'.format(data_file_path))
    app.config['data_file_path'] = data_file_path
    app.config['doc_ids'] = list(corpus_frame.index.values)
    app.config['titles'] = list(corpus_frame['title'])
    app.config['summaries'] = list(corpus_frame['summary'])
    app.config['links'] = list(corpus_frame['link'])
    app.config['tfidf'] = joblib.load(
        '{}/tfidf_sparse.txt'.format(data_file_path))
    app.config['nn_model'] = joblib.load(
        '{}/nn_model.txt'.format(data_file_path))
    app.config['vector_space'] = joblib.load(
        '{}/topic_space.txt'.format(data_file_path))
    app.config['kmodel'] = joblib.load(
        '{}/kmodel.txt'.format(data_file_path))
    app.config['cluster_reps'] = joblib.load(
        '{}/cluster_reps.txt'.format(data_file_path))
    # Constant that is used in the k(number of clusters) decision rule.
    app.config['k'] = len(app.config['kmodel'].cluster_centers_)
    app.run(debug=True)
