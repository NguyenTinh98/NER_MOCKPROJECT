def isNotSubword(x, idx, sub = '##'):
    if sub == '##':
        return sub not in x[idx] and idx < len(x) - 1 and sub not in x[idx+1]
    elif sub == '@@':
        return sub not in x[idx] and idx > 0 and sub not in x[idx-1]
    return sub in x[idx] and idx < len(x) - 1 and sub in x[idx+1]

def cutting_subword(X, y, size=256):
    res_X, res_y = [], []
    punct = '.!?'
    st = 0
    cur = 0
    
    while (st < len(X)-size):
        flag = True
        for i in range(st+size-1, st-1, -1):
            if X[i] in punct and y[i] == 'O':
                cur = i+1
                flag = False
                break
        if flag:
            for i in range(st+size-1, st-1, -1):
                if isNotSubword(X, i):
                    cur = i+1
                    if y[i] == 'O':
                        cur = i+1
                        break
        if st == cur:
            cur += size
            
        res_X.append(X[st: cur])
        res_y.append(y[st: cur])
        st = cur

    res_X.append(X[cur:])
    res_y.append(y[cur:])
    return res_X, res_y

############
def isNotSubword(x, idx, sub = '##'):
    if sub == '##':
        return sub not in x[idx] and idx < len(x) - 1 and sub not in x[idx+1]
    elif sub == '@@':
        return sub not in x[idx] and idx > 0 and sub not in x[idx-1]
    return sub in x[idx] and idx < len(x) - 1 and sub in x[idx+1]


def cutting_subword(X, sub = '##', size=256):
    res_X = []
    punct = '.!?'
    st = 0
    cur = 0
    while (st < len(X)-size):
        flag = True
        for i in range(st+size-1, st-1, -1):
            if X[i] in punct and isNotSubword(X, i, sub):
                cur = i+1
                flag = False
                break
        if flag:
            for i in range(st+size-1, st-1, -1):
                if isNotSubword(X, i, sub):
                    cur = i+1
                    break
        if st == cur:
            cur += size
        res_X.append(X[st: cur])
        st = cur
    res_X.append(X[cur:])
    return res_X

###############
import string
import unicodedata
def preprocessing_text(text):
    dictt = {'⁄': '/', '™': ' ', '‘': "'", '®': ' ', '×': ' ', '😀': ' ', '‑': ' - ', '́': ' ', '—': ' - ', '̣': ' ', '–': ' - ', '`': "'",\
             '“': '"', '̉': ' ','’': "'", '̃': ' ', '\u200b': ' ', '̀': ' ', '”': '"', '…': '...', '\ufeff': ' ', '″': '"'}
    text = unicodedata.normalize('NFKC', text)
    res = ''
    for i in text:
        if i.isalnum() or i in string.punctuation or i == ' ':
            res += i
        elif i in dictt:
            res += dictt[i]
    return res
