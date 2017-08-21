from flask import Flask, render_template, redirect, url_for, request
from flask_wtf import FlaskForm
from forms import ScatterGatherForm
import joblib
import numpy as np

app = Flask(__name__)
app.secret_key = 'v3rys3cr3t'
kmodel = joblib.load('data/kmodel.txt')
lda_model = joblib.load('data/lda_model.txt')
dictionary = joblib.load('data/dictionary.txt')
dist_space = joblib.load('data/dist_space.txt')
title_dict = joblib.load('data/title_dict.txt')
link_dict = joblib.load('data/link_dict.txt')
summary_dict = joblib.load('data/summary_dict.txt')

@app.route('/')
@app.route('/index', methods=['GET', 'POST'])


def index():
    sgform = ScatterGatherForm()
    cluster_reps = dict()
    for i in range(12):
        terms = []
        topic_id = kmodel.cluster_centers_[i].argsort()[::-1][0]
        term_ids = lda_model.get_topic_terms(topic_id, topn=3)
        for term_id in term_ids:
            terms.append(dictionary[term_id[0]])
            app.logger.debug(terms)
        cluster_reps[i] = terms
    if request.method == 'POST':
        doc_ids = []
        titles = []
        links = []
        summaries = []
        cluster_id = request.form['cluster_view']
        dist = dist_space[:, int(cluster_id)] # Distances of the documents form the cluster center.
        # Find the ids of the documents that belogn to this cluster.

        doc_ids = dist.argsort()[:50]
        for id in doc_ids:
            titles.append(title_dict[id])
            links.append(link_dict[id])
            summaries.append(summary_dict[id])
        return render_template('index.html', sgform=sgform, titles=titles,
                           links=links, summaries=summaries, cluster_reps=cluster_reps)
    return render_template('index.html', sgform=sgform, cluster_reps=cluster_reps) 

if __name__ == '__main__':
    app.run(debug=True, host='192.168.1.5')