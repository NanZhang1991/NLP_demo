## 启动redis
### 启动redis服务
redis-server
### 启动客户端
redis-cli

## 启动celery
celery -A app.celery_task  worker --loglevel=info

## 启动flask
python server.py
