from celery import Celery
from celery.result import AsyncResult
import time
from .main import function

celery = Celery('task',  broker='redis://localhost:6379/0', backend='redis://localhost:6379/1')
# celery.conf.update(app.config)


@celery.task
def get_start(data):
    taskId = data.get('taskId')
    fileId = data.get('fileId')
    details = data.get('data')
    function()
    '''
    此处省略耗时任务
    '''
    time.sleep(30) # 耗时任务

    return 'result:' + taskId

def get_status(task_id):
    task = AsyncResult(task_id, app=celery)
    # status = task.ready() 
    status = task.state # PENDING FAILURE SUCCESS RETRY STARTED
    return status
    

def get_result(task_id):
    task = AsyncResult(task_id, app=celery)
    return task.result

