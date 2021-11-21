import json
import pickle

def convert_jsondict2list(line):
    text = line['data']
    label = line['label']
    res = []
    ress = []
    cur = 0
    for s, e, l in label:
        if cur < s:
            res.append([cur, s-1, 'O'])
        res.append([s, e, l])
        cur = e + 1
    if cur < len(text):
        res.append([cur, len(text)-1, 'O'])
    for s, e, l in res:
        for i in text[s:e+1].strip().split():
            ress.append((i, l))
    return ress

def convert_json_admin(file):
    print('Convert admin jsonl')
    data = []
    with open (file, 'r') as f:
        for i in list(f):
            data.append(convert_jsondict2list((json.loads(i))))  
    return data

def convert_json_unknown(file):
    print('Convert unknow jsonl')
    data = []
    with open(file, 'r') as f:
        for i in list(f):
            text = json.loads(i)['data']
    for i in text.strip().split():
        data.append((i, 'O'))
    return data

def convert_data2pkl(data, file):
    with open(file, 'wb') as f:
        pickle.dump(data, f)
