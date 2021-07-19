## 启动celery
celery -A app.celery_task  worker --loglevel=info

## 启动flask
python server.py
