import json
import pickle

def sorted_idx(idx, isReverse = False):
    temp = [(u, v, a) for u, v, a in sorted(idx, key=lambda item: item[0], reverse = isReverse)]
    res = [temp[0]]
    for i in range(1, len(temp)):
        if isBelongRange(temp[i], temp[i-1]):
            res[-1] = temp[i]
        elif not isBelongRange(temp[i-1], temp[i]):
            res.append(temp[i])
    return res


def convert_jsondict2list(line):
    text = line['data']
    label = sorted_idx(line['label'])
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
    for i in range(len(res) - 1):
        if -res[i][1] + res[i+1][0] > 1:
            print(line['id']) 
    for s, e, l in res:
        for i in text[s:e].strip().split():
            ress.append((i, l))
    return ress

def convert_json_admin(file):
    print('Convert admin jsonl')
    data = []
    with open (file, 'r') as f:
        for i in list(f):
            temp = convert_jsondict2list((json.loads(i)))
            if len(temp) > 0:
                data.append(temp)  
    return data

def convert_json_unknown(file):
    print('Convert unknow jsonl')
    data = []
    with open(file, 'r') as f:
        for i in list(f):
            text = json.loads(i)['data']
            temp = []
            for i in text.strip().split():
                temp.append((i, 'O'))
            if len(temp) > 0:
                data.append(temp)
    return data

def convert_data2pkl(data, file):
    with open(file, 'wb') as f:
        pickle.dump(data, f)
        
        
        
## pipeline convert jsonl to pkl
## path to file admin jsonl and unknown jsonl
admins = 'real_final/admin.jsonl'
unknows = 'real_final/unknown.jsonl'
total_data = []
file_pkl = 'final1.pkl'
total_data += convert_json_admin(i)
total_data += convert_json_unknown(i)
convert_data2pkl(total_data, file_pkl)
