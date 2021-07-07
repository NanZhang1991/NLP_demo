import sys
from app import app

if __name__ == '__main__':
    if sys.platform =='linux':
        app.run(host="0.0.0.0", port=8010, processes=1, threaded=False, debug=True)
    else:
        app.run(host="0.0.0.0", port=8010, threaded=False, debug=True)