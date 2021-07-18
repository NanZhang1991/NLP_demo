import sys
from flask import Flask, request, make_response, jsonify
from celery import Celery
from flask import Flask
from celery import Celery
from celery.result import AsyncResult

app = Flask(__name__)
app.config['CELERY_BROKER_URL'] = 'redis://localhost:6379/0'
app.config['CELERY_RESULT_BACKEND'] = 'redis://localhost:6379/0'

celery = Celery(app.name, broker=app.config['CELERY_BROKER_URL'], backend=app.config['CELERY_RESULT_BACKEND'])
celery.conf.update(app.config)

@celery.task
def get_start(url):
    return '1234'

@app.route('/api/task_start', methods=['POST'])
def task_start():
 request_data = request.get_json(force=True)
 task_data = {}
 task_data['url'] = request_data.get("target")
 rsp = get_start.delay(task_data['url'])
 task_id = rsp.task_id
 return make_response(jsonify(url=task_data['url'], task_id=task_id))


@app.route('/api/task_start', methods=['GET'])
def task_status():
    pass

@app.route('/api/task_start', methods=['GET'])

def task_result():
    pass


@celery.task
def get_start(url):
    return '1234' 

if __name__ == '__main__':
    if sys.platform =='linux':
        app.run(host="0.0.0.0", port=8010, processes=1, threaded=False, debug=True)
    else:
        app.run(host="0.0.0.0", port=8010, threaded=False, debug=True)