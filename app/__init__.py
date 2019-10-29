from flask import Flask

app = Flask(__name__)

from app import routes  # nopep8
from app import task_routes  # nopep8
