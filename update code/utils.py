from spacy import displacy
from sklearn.metrics import *
import string
import unicodedata
import re
import numpy as np
import pickle

COLORS ={
    'EMAIL':'#FDEE00',
    'ADDRESS':'#C32148',
    'PERSON':'#FE6F5E',
    'PHONENUMBER': '#9F8170',
    'MISCELLANEOUS':'#007BA7',
    'QUANTITY':'#D891EF',
    'PERSONTYPE':'#FF91AF',
    'ORGANIZATION':'#3DDC84',
    'PRODUCT':'#FBCEB1',
    'SKILL':'#B0BF1A',
    'IP':'#703642',
    'LOCATION':'#C0E8D5',
    'DATETIME':'aqua',
    'EVENT':'darkorange',
    'URL':'#BD33A4'
}

NER_COLOR = list(COLORS.keys())
OPTIONS = {'ents': NER_COLOR, 'colors': COLORS}


def isNotSubword(x, idx, sub = '##'):
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
##########################################################
def merge_subtags(tokens, tags_predict):
    tags = []
    tests = []
    for index in range(len(tokens)):
        if len(tests) == 0:
            if "▁" in tokens[index]:
                tests.append(tokens[index][1:])
            else:
                tests.append(tokens[index])
            tags.append(tags_predict[index])
        elif "▁" in tokens[index]:
            tests.append(tokens[index][1:])
            tags.append(tags_predict[index])
        else:
            tests[-1] = tests[-1] + tokens[index]
    return tests, tags

def merge_subtags_3column(tokens, tags_true, tags_predict):
    tags = []
    tests = []
    trues = []
    for index in range(len(tokens)):
        if len(tests) == 0:
            if "▁" in tokens[index]:
                tests.append(tokens[index][1:])
            else:
                tests.append(tokens[index])
            tags.append(tags_predict[index])
            trues.append(tags_true[index])
        elif "▁" in tokens[index] or ('<' in tokens[index] and '>' in tokens[index]):
            tests.append(tokens[index][1:])
            tags.append(tags_predict[index])
            trues.append(tags_true[index])
        else:
            tests[-1] = tests[-1] + tokens[index]
    return tests, trues, tags

def merge_subtags_4column(tokens, tags_predict, sm):
    tags = []
    tests = []
    sms = []
    temp = []
    for index in range(len(tokens)):
        if len(tests) == 0:
            if "▁" in tokens[index]:
                tests.append(tokens[index][1:])
            else:
                tests.append(tokens[index])
            tags.append(tags_predict[index])
            sms.append(sm[index])
        elif "▁" in tokens[index] or "</s>" in tokens[index]:
            tests.append(tokens[index][1:])
            tags.append(tags_predict[index])
            temp.append(sm[index])
            sms.append(np.mean(temp, axis=0))
            temp = []
        else:
            tests[-1] = tests[-1] + tokens[index]
            temp.append(sm[index])
    return tests, tags, sms
############################################################
def visualize_spacy(arr):
    if len(arr) < 1:
        return None
    text = ' '.join([i for i, j in arr])
    pos = 0
    start_end_labels = []
    for word, tag in arr:
        if len(start_end_labels) > 0 and tag == start_end_labels[-1][2]:
            temp = [start_end_labels[-1][0], pos+len(word), tag]
            start_end_labels[-1] = temp.copy()
        else:
            temp = [pos, pos+len(word), tag]
            start_end_labels.append(temp)
        pos += len(word) + 1
        
    ex = [{'text': text, 'ents': [{'start': x[0], 'end': x[1], 'label': x[2]} for x in start_end_labels if x[2]!= 0]}]
    return displacy.render(ex, manual=True, jupyter=False, style='ent', options = OPTIONS)#page=True
#########################################
def softmax(arr):
    return np.exp(arr) / sum(np.exp(arr))

def write_pickle(dt, path):
  with open(path, 'wb') as f:
    pickle.dump(dt, f)

def read_pickle(file):
  with open(file, 'rb') as f:
    return pickle.load(f)
#=========convert data for auto-labelling==============
def preprocessing_text2(text):
    dictt = {'™': ' ', '‘': "'", '®': ' ', '×': ' ', '😀': ' ', '‑': ' - ', '́': ' ', '—': ' - ', '̣': ' ', '–': ' - ', '`': "'",\
    '“': '"', '̉': ' ','’': "'", '̃': ' ', '\u200b': ' ', '̀': ' ', '”': '"', '…': '...', '\ufeff': ' ', '″': '"'}
    text = text.split('\n')
    text = ' '.join([i.strip() for i in text if i!=''])
    text = unicodedata.normalize('NFKC', text)
    res = ''
    for i in text:
        if i.isalnum() or i in string.punctuation or i == ' ':
            res += i
        elif i in dictt:
            res += dictt[i]
    text = preprocess_data(res)
    return text

def read_txt(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.readlines()

def write_txt(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        for i in data:
            f.write(i + '\n')

def convertfile2doccano(path, path2):
    res = []
    data = read_txt(path)
    for line in data:
        res.append(preprocessing_text2(line.strip()))
        write_txt(res, path2)


# #
# 1. Merge subtags 2column
# #

def merge_subtags(tokens, tags_predict, model_name):
    tags = []
    tests = []
    if 'mbert' in model_name:
        for index in range(len(tokens)):
            if "##" not in tokens[index]:
                tags.append(tags_predict[index])
                tests.append(tokens[index])
            else:
                tests[-1] = tests[-1] + tokens[index].replace("##","")
    elif 'phobert' in model_name:
        for index in range(len(tokens)):
            if len(tests) == 0:
                tests.append(tokens[index])
                tags.append(tags_predict[index])
            elif "@@" in tests[-1]:
                tests[-1] = tests[-1][:-2] + tokens[index]
            else:
                tests.append(tokens[index])
                tags.append(tags_predict[index])
    elif 'xlmr' in model_name:
        for index in range(len(tokens)):
            if len(tests) == 0:
                if "▁" in tokens[index]:
                    tests.append(tokens[index][1:])
                else:
                    tests.append(tokens[index])
                tags.append(tags_predict[index])
            elif "▁" in tokens[index]:
                tests.append(tokens[index][1:])
                tags.append(tags_predict[index])
            else:
                tests[-1] = tests[-1] + tokens[index]
    return tests, tags

# #
# 2. Merge subtags 3column
# #
def merge_subtags_3column(tokens, tags_true, tags_predict, model_name):
    tags = []
    tests = []
    trues = []
    if 'mbert' in model_name:
        for index in range(len(tokens)):
            if "##" not in tokens[index]:
                tags.append(tags_predict[index])
                tests.append(tokens[index])
                trues.append(tags_true[index])
            else:
                tests[-1] = tests[-1] + tokens[index].replace("##","")
    elif 'phobert' in model_name:
        for index in range(len(tokens)):
            if len(tests) == 0:
                tests.append(tokens[index])
                tags.append(tags_predict[index])
                trues.append(tags_true[index])
            elif "@@" in tests[-1]:
                tests[-1] = tests[-1][:-2] + tokens[index]
            else:
                tests.append(tokens[index])
                tags.append(tags_predict[index])
                trues.append(tags_true[index])
    elif 'xlmr' in model_name:
        for index in range(len(tokens)):
            if len(tests) == 0:
                if "▁" in tokens[index]:
                    tests.append(tokens[index][1:])
                else:
                    tests.append(tokens[index])
                tags.append(tags_predict[index])
                trues.append(tags_true[index])
            elif "▁" in tokens[index]:
                tests.append(tokens[index][1:])
                tags.append(tags_predict[index])
                trues.append(tags_true[index])
            else:
                tests[-1] = tests[-1] + tokens[index]
    return tests, trues, tags