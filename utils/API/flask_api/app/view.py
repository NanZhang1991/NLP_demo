
import requests
import os
from pathlib import Path
import sys
import json
from ast import literal_eval
import pandas as pd
from flask import Blueprint, request, send_file, send_from_directory, make_response, jsonify, url_for, current_app
from .common.log import logger
from .common.dataload import load_json
from .common.exception import CustomException
from .common.fileCheck import allowed_file
from .config.config import log_dir

logger = logger(os.path.join(log_dir, 'view.log'), __name__)
appName = Blueprint("docx/appName", __name__, url_prefix='/docx/appName')

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

@appName.route('/spo/comment', methods=['POST'], strict_slashes=False)
def spo_comment():
    data = json.loads(request.get_data())
    taskId = data.get('taskId')
    fileId = data.get('fileId')
    details = data.get('data')
    spoIdList = data.get('spoIdList')
    logger.info(f"request data---------\n taskId:{taskId}, \n spoIdList--{spoIdList}")# 
    message = {}

    params = {'fileId': fileId}
    getFileUrlText = requests.get(downloadfileIp, params=params).text
    logger.info(f'getFileUrlText-----------{getFileUrlText}')
    getFileUrlDict = json.loads(getFileUrlText)
    if getFileUrlDict.get('data'):
        fileUrl = getFileUrlDict.get('data').get('fileUrl')
        message['getFileMsg'] = getFileUrlDict.get('msg')

        if os.path.exists(taskCsvPath):
            task_df = pd.read_csv(taskCsvPath, encoding='utf-8', dtype={'code':str})
        else:
            task_df = pd.DataFrame(columns=['taskId', 'data','msg'])
        idx  = task_df.shape[0]   

        if allowed_file(fileUrl) =='docx':
            content = requests.get(fileUrl).content
            fn = fileId + '.docx'
            input_fp = os.path.join(UPLOAD_FOLDER, fn)
            input_fp = Path(input_fp).as_posix()
            with open(input_fp, 'wb') as f:
                f.write(content)
            message['comment_message'], out_fn = Comment.review_comment(input_fp, details,  OUTPUT_FOLDER)

            uploadDict = {'userId':'001', 'userName':'review'} #
            uploadResponseText = requests.post(uploadFileIp, data=uploadDict, files={"file": open(out_fn, "rb")}).text
            uploadReponseDict = literal_eval(uploadResponseText)
            logger.info(f'uploadResponseText--{uploadResponseText}')
            fileBucketId = uploadReponseDict.get('data').get('fileBucketId')
            message["uploadFileMSg"] = uploadReponseDict.get('msg')

            code= 200
            task_df.loc[idx ,"taskId"] = taskId
            task_df.loc[idx ,"code"] = code
            task_df.loc[idx, 'data'] = json.dumps({'fileIdList':[fileBucketId]}, ensure_ascii=False)
            task_df.loc[idx ,"msg"]= json.dumps(message, ensure_ascii=False)
            task_df.to_csv(taskCsvPath, index=False, encoding='utf_8_sig')

        else:
            message['comment_message'] = 'File format error'
            code = 205 
    else:
        code = getFileUrlDict.get('code')
        message['getFileMsg'] = "File not found"

    logger.info(f'message ----:{message}\n')
    result = jsonify({'code':code, 'msg':message})
    return  result

@appName.route('/query', methods=['GET'], strict_slashes=False)
def query_task():
    taskId = request.args.get('taskId')
    try:
        if os.path.exists(taskCsvPath):
            task_df = load_task(taskCsvPath)
            taskIndex = task_df['taskId'][task_df['taskId']==taskId].index.to_list()
            if taskIndex:
                idx = taskIndex[0]
                result = task_df.loc[idx, ['code', 'data', 'msg']]
                result_str = result.to_json(orient='columns', force_ascii=False, indent=4)
                result.to_json(OUTPUT_FOLDER + '/' + taskId + '.json', orient='columns', force_ascii=False, indent=4)
                return result_str
            else:
                raise CustomException('The taskId is not exist')
        else:
            raise CustomException('Task file not found')
    except Exception as e:
        logger.info(f"Error:{e}\n")
        return jsonify({'code':201, 'data':{'fileIdList':None,}, 'msg':{"Error":str(e)}})
    

    
    

