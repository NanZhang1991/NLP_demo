
import requests
import os
from pathlib import Path
import sys
import json
from ast import literal_eval
import pandas as pd
from flask import Blueprint, request, send_file, send_from_directory, make_response, jsonify, url_for, current_app
from .common.log import logger
from .common.dataload import load_json, load_task
from .common.exception import CustomException
from .common.fileCheck import allowed_file
from .config.config import log_dir
from .celery_task import get_start, get_status, get_result


logger = logger(os.path.join(log_dir, 'view.log'), __name__)
appName = Blueprint("/api", __name__, url_prefix='/api')

json_path = 'app/config/view_config.json'
view_config = load_json(json_path)
UPLOAD_FOLDER = view_config.get('UPLOAD_FOLDER')
OUTPUT_FOLDER = view_config.get('OUTPUT_FOLDER')
downloadfileIp = view_config.get('downloadFileIp') 
uploadFileIp = view_config.get('uploadFileIp')
taskCsvPath = view_config.get('taskCsvPath')

@appName.route('/')
def test():
    return '200' 

@appName.route('/task_start', methods=['POST'], strict_slashes=False)
def task_start():
    data = request.get_json(force=True)
    # taskId = data.get('taskId')
    # fileId = data.get('fileId')
    # details = data.get('data')
    logger.info(f"request data---------\n  data:{data}")
    task_data = {}
    #异步任务
    task = get_start.apply_async([data]) # 以列表形式传参
    task_id = task.task_id
    print('---------------', task_id)
    return make_response(jsonify(code=200, task_id=task_id, msg='success'))


@appName.route('/task_status', methods=['GET'])
def task_status():
    task_id = request.args.get('task_id')
    status = get_status(task_id)
    return make_response(jsonify(task_id=task_id, status=status))   

@appName.route('/task_result', methods=['GET'])
def task_result():
    task_id = request.args.get('task_id')
    result = get_result(task_id)
    return make_response(jsonify(taskid=task_id, result=result))

    

    
    

