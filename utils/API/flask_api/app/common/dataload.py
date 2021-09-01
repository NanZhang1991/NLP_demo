import json
from numpy import int8
import pandas as  pd
from ast import literal_eval

def load_json(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        dic = json.load(f)
    return dic

def load_task(taskXlsxPath):
    task_df = pd.read_csv(taskXlsxPath, encoding='utf-8', dtype={'code':int})
    task_df['data'] = task_df['data'].apply(literal_eval)
    task_df['msg'] = task_df['msg'].apply(literal_eval)
    return task_df

if __name__=='__main__':
    json_path = 'doc_review/config/view_config.json'
    print(load_json(json_path))