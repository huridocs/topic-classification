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


@app.route('/embed/<uuid:id>', methods=['GET'])
def embed():
    e = embedder.Embedder()
    e.embed()
    return "Embedding goes here"
