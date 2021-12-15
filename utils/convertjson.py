import json
import pickle

def isBelongRange(a, b):
     return b[0] >= a[0] and b[1] <= a[1]

import json
import pickle

def sorted_idx(idx, isReverse = False):
    if len(idx) == 0:
        return idx
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
            res.append([cur, s, 'O'])
        res.append([s, e, l])
        cur = e
    if cur < len(text):
        res.append([cur, len(text), 'O'])
    
    ## check res append abide by after - before
    for i in range(len(res) - 1):
        if -res[i][1] + res[i+1][0] > 0:
            print(line['id']) 
    
    for s, e, l in res:
        for i in text[s:e].strip().split():
            ress.append((i, l))
    
    ## check text_data before and after
    if len(ress) > 0:
        a, _ = list(zip(*ress))
        if ' '.join(a) != text:
            print(text)
            print(res)
            print(' '.join(a))
            print(len(text), len(' '.join(a)))
    return ress

def convert_json_file(file):
    print('Convert admin jsonl')
    data = []
    with open(file, 'r', encoding='utf-8') as f:
        for i in list(f):
            temp = convert_jsondict2list((json.loads(i)))
            if len(temp) > 0:
                data.append(temp)  
    return data

def convert_data2pkl(data, file):
    with open(file, 'wb') as f:
        pickle.dump(data, f)        
        
        
## pipeline convert jsonl to pkl
## path to file admin jsonl and unknown jsonl
## pipeline convert jsonl to pkl
admins = 'admin.jsonl'
#unknows = '7900_299831/unknown.jsonl'
total_data = []
file_pkl = 'final_augment.pkl'
total_data += convert_json_file(admins)
#total_data += convert_json_file(unknowns)
convert_data2pkl(total_data, file_pkl)

