from flask import Flask
from flask_cors import CORS
from . import view

app = Flask(__name__)

app.config["JSON_AS_ASCII"]=False
CORS(app, resources=r'/*')

@app.route('/')
def test():
    return 'Port test successful!'

app.debug = True
with app.app_context():
    # appName 根据自己实际的路由注册
    app.register_blueprint(view.appName) #
