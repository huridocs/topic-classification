from typing import Any

from flask import Blueprint, abort
from flask import current_app as app
from flask import jsonify, request

from app import tasks

task_bp = Blueprint('task_bp', __name__)


@task_bp.route('/task', methods=['GET'])
def get_task() -> Any:
    name = (request.args['name']
            if 'name' in request.args else request.get_json()['name'])
    t = tasks.GetTask(name)
    if not t:
        abort(404)
    return jsonify({
        'status': t.status,
        'state': t.state,
        'start_time': t.start_time,
        'end_time': t.end_time
    })


@task_bp.route('/task', methods=['POST'])
def push_task() -> Any:
    data = request.get_json()
    if 'base_classifier_dir' not in data:
        data['base_classifier_dir'] = app.config['BASE_CLASSIFIER_DIR']
    p = tasks.GetProvider(data['provider'])
    if not p or not data['name']:
        abort(400)
    t = tasks.GetOrCreateTask(data['name'], p(data))
    if not t:
        abort(400)
    if not t.thread:
        t.Start()
    return jsonify({
        'status': t.status,
        'state': t.state,
        'start_time': t.start_time,
        'end_time': t.end_time
    })


@task_bp.route('/task', methods=['DELETE'])
def delete_task() -> Any:
    data = request.get_json()
    t = tasks.GetTask(data['name'])
    if not t:
        abort(404)
    if t.thread:
        t.Stop(join=False)
    return jsonify({
        'status': t.status,
        'state': t.state,
        'start_time': t.start_time,
        'end_time': t.end_time
    })
