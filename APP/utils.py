from spacy import displacy
from sklearn.metrics import *
import string
import unicodedata
import re

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

############################################################
def preprocess_data(sent):
    sent = '<s> ' + sent + ' </s>'
    #sent = '$ ' + sent
    sent = handle_bracket(sent)
    sent = re.sub(' +', ' ', sent)
    sent_out = ""
    parts = sent.split()
    sent_out = ' '.join([handle_character(i)[0] for i in parts])
    return sent_out

def process_unk(tokenizer, sq):
    temp = []
    for i in sq.split():
        if ['[UNK]'] == tokenizer.tokenize(i):
            temp.append(i[0]+i[1:].lower())
        else:
            temp.append(i)
    return ' '.join(temp)

def preprocessing_text(tokenizer,text):
    dictt = {'™': ' ', '‘': "'", '®': ' ', '×': ' ', '😀': ' ', '‑': ' - ','́': ' ', '—': ' - ', '̣': ' ', '–': ' - ', '`': "'",\
             '“': '"', '̉': ' ','’': "'", '̃': ' ', '\u200b': ' ', '̀': ' ', '”': '"', '…': '...', '\ufeff': ' ', '″': '"'}
    text = text.split('\n')
    text = [i.strip()  for i in text if i!='']
    out = ""
    for i in range(1,len(text)+1):
        out += text[i-1]+' .</s> '
    text = unicodedata.normalize('NFKC', out)
    res = ''
    for i in text:
        if i.isalnum() or i in string.punctuation or i == ' ':
            res += i
        elif i in dictt:
            res += dictt[i]
    text = preprocess_data(res)
    return process_unk(tokenizer,text)

def handle_bracket(test_str):
    res = re.findall(r'(\(|\[|\"|\'|\{)(.*?)(\}|\'|\"|\]|\))', test_str)
    if len(res) > 0:
        for r in res:
            sub_tring = "".join(r)
            start_index = test_str.find(sub_tring)
            end_index = start_index + len(r[1])
            test_str = test_str[: start_index+ 1] + " " + test_str[start_index+ 1:]
            test_str = test_str[: end_index + 2] + " " + test_str[end_index + 2:]
    return test_str

def handle_character(sub_string):
    char_end = [".", ",", ";", "?", "+", ":" ]
    count = 1
    for index in reversed(range(len(sub_string))):
        c = sub_string[index]
        if c not in char_end:
            break
        elif c in char_end:
            if sub_string[index -1] not in char_end:
                sub_string = sub_string[:index] + " " + sub_string[index:]
                count = 2
            break
    return sub_string, count
#################################
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