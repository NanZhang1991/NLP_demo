
import pandas as pd

ALLOWED_EXTENSIONS = set(['docx'])

def allowed_file(fn):
    if '.' in fn and fn.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS:
        return fn.rsplit('.', 1)[1]
    else:
        return False