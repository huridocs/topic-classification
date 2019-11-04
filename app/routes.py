import json

from app import app
from app import classifier
from app import embedder
from app import tasks

from flask import request


@app.route('/')
@app.route('/index')
def index():
    return "Hello, World!"


@app.route('/classify', methods=['POST'])
def classify():
    # request.args: {'classifier': 'bert_for_countries'}
    # request.form: {'seq1', 'seq2', 'seq3'}
    error = None
    c = classifier.Classifier()
    c.classify()
    return "Classification goes here"


@app.route('/embed', methods=['POST'])
def make_embed():
    # request.args: {'bert': 'bert_uncased_L-12_H-768_A-12/1'}
    error = None
    e = embedder.Embedder(request.form['bert'])
    matrix = e.GetEmbedding(request.form['seq'])
    return str(len(matrix))


@app.route('/embed/<uuid:id>', methods=['GET'])
def fetch_embed(id):
    e = embedder.Embedder()
    e.embed()
    return "Embedding goes here"
