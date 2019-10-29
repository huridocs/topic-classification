from flask import request
from app import app
from app import tasks
import json


@app.route('/')
@app.route('/index')
def index():
    return "Hello, World!"
