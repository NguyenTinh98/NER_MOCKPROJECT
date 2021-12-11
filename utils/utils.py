########################################################################################################################################
from spacy import displacy
from sklearn.metrics import *
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

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


NER = list(COLORS.keys())



OPTIONS = {'ents': NER, 'colors': COLORS}
    
## visualize result
## input: predict format [(word, tag)]

def bertvisualize(arr):
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
        
    ex = [{'text': text, 'ents': [{'start': x[0], 'end': x[1], 'label': x[2]} for x in start_end_labels if x[2]!= 'O']}]
    displacy.render(ex, manual=True, jupyter=True, style='ent', options = OPTIONS )
    #return ex

########################################################################################################################################
def preprocessing_text(tokenizer,text):
    dictt = {'™': ' ', '‘': "'", '®': ' ', '×': ' ', '😀': ' ', '‑': '-', '́': ' ', '—': '-', '̣': ' ', '–': '-', '`': "'",\
    '“': "'", '̉': ' ','’': "'", '̃': ' ', '\u200b': ' ', '̀': ' ', '”': "'", '…': '...', '\ufeff': ' ', '″': "'"}
    for i in dictt:
        text = text.replace(i, dictt[i])
    text = preprocess_data(text)
    return process_unk(tokenizer,text)

def process_unk(tokenizer, sq):
    temp = []
    for i in sq.split():
        if ['[UNK]'] == tokenizer.tokenize(i):
            temp.append(i[0]+i[1:].lower())
        else:
            temp.append(i)
    return ' '.join(temp)

import re


def handle_bracket(test_str):
  res = re.findall(r'(\(|\[|\"|\'|\{)(.*?)(\}|\'|\"|\]|\))', test_str)
  # print(res)
  if len(res) > 0:
    for r in res:
      sub_tring = "".join(r)
      start_index = test_str.find(sub_tring)
      end_index = start_index + len(r[1])
      test_str = test_str[: start_index+ 1] + " " + test_str[start_index+ 1:]
      test_str = test_str[: end_index + 2] + " " + test_str[end_index + 2:]
      # test_str = 
  return test_str

def handle_character(sub_string):
  char_end = [".", ",", ";", "?", "+", ":" ]
  count = 1
  for index in reversed(range(len(sub_string))):

    
    # print(index)
    c = sub_string[index]
    # print(index, c)

    #check black token

    if c not in char_end:
      break
    
    elif c in char_end:
      # print(sent[index -1] )
      if sub_string[index -1] not in char_end:
        # print(sub_string[index -1])
        sub_string = sub_string[:index] + " " + sub_string[index:]
        count = 2
        break

  return sub_string, count

def preprocess_data(sent):
  sent = handle_bracket(sent)
  sent = re.sub(' +', ' ', sent)
  sent_out = ""
  parts = sent.split()

  for index in range(len(parts)):
    word_space = parts[index]
    # print(word_space)

    sub_string_handeled, _ = handle_character(word_space)
    
    if index != len(parts) - 1:
      sent_out +=  sub_string_handeled + " "
    else:
      sent_out += sub_string_handeled
  return sent_out

########################################################################################################################################
# def isNotSubword(x, idx, sub = '##'):
#     return sub not in x[idx] and idx > 0 and idx < len(x) - 1 and sub not in x[idx-1] and sub not in x[idx+1]
def isNotSubword(x, idx, sub = '##'):
    if sub == '##':
        return sub not in x[idx] and idx < len(x) - 1 and sub not in x[idx+1]
    elif sub == '@@':
        return sub not in x[idx] and idx > 0 and sub not in x[idx-1]
    return sub in x[idx] and idx < len(x) - 1 and sub in x[idx+1]
def cutting_subword(X, sub = '##', size=256):
    res_X = []
    punct = '.!?;'
    st = 0
    cur = 0
    while (st < len(X)-size):
        flag = True
        for i in range(st+size-1, st-1, -1):
            if X[i] in punct:
                cur = i+1
                flag = False
                break
        if flag:
            for i in range(st+size-1, st-1, -1):
                if isNotSubword(X, i, sub):
                    cur = i+1
                    break
        res_X.append(X[st: cur])
        st = cur
    res_X.append(X[cur:])
    return res_X
########################################################################################################################################

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


def merge_logits_subtags(tokens, tags_predict, out_logits, model_name):
    tags = []
    tests = []
    logits = []
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
                logits.append(out_logits[index])
            elif "▁" in tokens[index]:
                tests.append(tokens[index][1:])
                tags.append(tags_predict[index])
                logits.append(out_logits[index])
            else:
                tests[-1] = tests[-1] + tokens[index]
    return tests, tags, logits


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
########################################################################################################################################
def convert_spanformat(arr):
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

    res = dict()   
    for s, e, l in start_end_labels:
        if l != 'O':
            if l not in res:
                res[l] = [(s, e)]
            else:
                res[l].append((s, e))
    return res
 
def compare_span(span1, span2, res, strict= True):
    all_labels = set(list(span1.keys()) + list(span2.keys()))
    for l in all_labels:
        if l not in res:
            res[l] = [0, 0, 0, 0]
        if l not in span1:
            res[l][3] += len(span2[l])
            continue
        if l not in span2:
            res[l][2] += len(span1[l])
            continue
        res[l][2] += len(span1[l])
        res[l][3] += len(span2[l])
        for s, e in span1[l]:
            for s1, e1 in span2[l]:
                temp0, temp1 = iou_single(s, e, s1, e1)
                if strict:
                    temp0, temp1 = int(temp0), int(temp1)
                res[l][0] += temp0
                res[l][1] += temp1
    return res
 
def iou_single(s1, e1, s2, e2):
    smax = max(s1, s2)
    emin = min(e1, e2)
    return max(0, emin - smax) / (e1 - s1) if e1 - s1 > 0 else 0, max(0, emin - smax) / (e2 - s2) if e2 - s2 > 0 else 0
 
# (token - True - pred) 
# [[ ],[ ]]           
def span_f1(arr, labels = None, strict=True, digit=4):
    all_labels = set()
    dictt = dict()
    for ar in arr:
        text, gt, pred = list(zip(*ar))
        gtSpan = convert_spanformat(list(zip(text, gt)))
        predSpan = convert_spanformat(list(zip(text, pred)))
        dictt = compare_span(predSpan, gtSpan, dictt, strict)

        all_labels.update(list(gtSpan.keys()))
    classfication_rp = dict()
    # print(dictt)
    f1_avg = 0
    if labels is None:
        labels = all_labels
    for i in labels:
        precision = dictt[i][0] / dictt[i][2] if dictt[i][2] > 0 else 0
        recall = dictt[i][1] / dictt[i][3] if dictt[i][3] > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
        classfication_rp[i] = {'precision': round(precision, digit), 'recall': round(recall, digit), 'f1': round(f1, digit)}
        f1_avg += f1
    return f1_avg / len(labels), classfication_rp

########################################################################################################################################
def summary_result(y_true, y_pred, is_show = False):
    LABELS = list(set(y_true))
    cm = confusion_matrix(y_true, y_pred, labels = LABELS)
    rp = classification_report(y_true, y_pred, labels = LABELS, digits=3)
    print(rp)
    if is_show == True:
        df_cm = pd.DataFrame(cm, LABELS, LABELS)
        df_cm.index.name = 'Actual'
        df_cm.columns.name = 'Predicted'
        plt.figure(figsize = (10,7))
        sn.heatmap(df_cm, annot=True, fmt='d', cmap='Blues')
        plt.show()

########################################################################################################################################
import pickle
def write_pickle(dt, path):
  with open(path, 'wb') as f:
    pickle.dump(dt, f)

def read_pickle(file):
  with open(file, 'rb') as f:
    return pickle.load(f)

import json
def write_json(dt, path):
  with open(path, 'w', encoding='utf-8') as f:
    json.dump(dt, f)

def read_json(path):
  with open(path, 'r', encoding='utf-8') as f:
    return json.load(f)
def convert_pkl2txt(data,path):
    with open(path, 'w', encoding='utf-8') as f:
        for sq in data:
            for txt in sq:
                ww, tt = txt
                f.write(ww+'\t'+tt)
                f.write('\n')
            f.write('\n')

def convert_to_BItag(sq):
    out = []
    for data in sq :
        newtag= []
        for i in range(len(data)):
            if data[i][1] != 'O':
                if i == 0 or data[i-1][1] != data[i][1]:
                    temp = 'B-' + data[i][1]
                elif i == len(data)-1 or  data[i+1][1] != data[i][1] :
                    temp = 'I-' + data[i][1]
                else:
                    temp = 'I-' + data[i][1]
            else:
                temp = data[i][1]
            newtag.append((data[i][0],temp))
        out.append(newtag)
    return out
########################################################################################################################################
def show_span_f1(dic):
    index = []
    da = []
    for tag, detail in dic.items():
        index.append(tag)
        da.append(detail)
    df = pd.DataFrame(da)
    df = df.set_index([pd.Index(index)])
    return df

################################################################################
#hậu xử lý
def quet_dinh_nhan(token):
  all_freq = {}
  word = token[0]
  # print(token[0], token[1])
  if token[0][-1] in [",", ".", ";", "?", "!"] and token[1][-1] == "O" and len(token[1]) > 1:
    ended = (token[0][-1], token[1][-1])
    for i in token[1][:-1]:
      if i in all_freq:
          all_freq[i] += 1
      else:
          all_freq[i] = 1
    res = max(all_freq, key = all_freq.get)
    return [(token[0][:-1], res), ended]
  else:
    for i in token[1]:
        if i in all_freq:
            all_freq[i] += 1
        else:
            all_freq[i] = 1
    res = max(all_freq, key = all_freq.get)
    return [(token[0], res)]


import re

def is_URL(token):
    index = 0
    indexs = []
    for word in token.split(" "):
        # print(word)
        domain = re.findall(r'\b((?:https?://)?(?:(?:www\.)?(?:[\da-z\.-]+)\.(?:[a-z]{2,6})|(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)|(?:(?:[0-9a-fA-F]{1,4}:){7,7}[0-9a-fA-F]{1,4}|(?:[0-9a-fA-F]{1,4}:){1,7}:|(?:[0-9a-fA-F]{1,4}:){1,6}:[0-9a-fA-F]{1,4}|(?:[0-9a-fA-F]{1,4}:){1,5}(?::[0-9a-fA-F]{1,4}){1,2}|(?:[0-9a-fA-F]{1,4}:){1,4}(?::[0-9a-fA-F]{1,4}){1,3}|(?:[0-9a-fA-F]{1,4}:){1,3}(?::[0-9a-fA-F]{1,4}){1,4}|(?:[0-9a-fA-F]{1,4}:){1,2}(?::[0-9a-fA-F]{1,4}){1,5}|[0-9a-fA-F]{1,4}:(?:(?::[0-9a-fA-F]{1,4}){1,6})|:(?:(?::[0-9a-fA-F]{1,4}){1,7}|:)|fe80:(?::[0-9a-fA-F]{0,4}){0,4}%[0-9a-zA-Z]{1,}|::(?:ffff(?::0{1,4}){0,1}:){0,1}(?:(?:25[0-5]|(?:2[0-4]|1{0,1}[0-9]){0,1}[0-9])\.){3,3}(?:25[0-5]|(?:2[0-4]|1{0,1}[0-9]){0,1}[0-9])|(?:[0-9a-fA-F]{1,4}:){1,4}:(?:(?:25[0-5]|(?:2[0-4]|1{0,1}[0-9]){0,1}[0-9])\.){3,3}(?:25[0-5]|(?:2[0-4]|1{0,1}[0-9]){0,1}[0-9])))(?::[0-9]{1,4}|[1-5][0-9]{4}|6[0-4][0-9]{3}|65[0-4][0-9]{2}|655[0-2][0-9]|6553[0-5])?(?:/[\w\.-]*)*/?)\b', word)
        
        if len(domain) != 0:
            index_start_domain = word.find(domain[0]) + index
            if word.find(domain[0]) == 0:
                index_end_domain = index_start_domain + len(word)
            else:
                index_end_domain = index_start_domain + len(domain[0])
            indexs.append((index_start_domain, index_end_domain))
        index += len(word) + 1
    return indexs

def is_Email(token):
    index = 0
    indexs = []
    for word in token.split(" "):
        # print(word)
        emails = re.findall(r"[\w.+-]+@[\w-]+\.[\w.-]+", word)
        # print(emails)
        if len(emails) != 0:
            index_start_email = word.find(emails[0]) + index
            
            index_end_email = index_start_email + len(emails[0])
            
            indexs.append((index_start_email, index_end_email))
        index += len(word) + 1
    return indexs

def preprocess_email_url(datas):
  datas_trained = []
  for i in range(len(datas)):
    data = datas[i]

    if data[1] == 'EMAIL':
      check = is_Email(data[0])
      if len(check) == 0:
        data = (data[0], 'O')
    
    if data[1] != 'EMAIL' and  data[1] != 'URL': #(url, org, loc, o,.....)
      check = is_Email(data[0])
      if len(check) > 0:
        data = (data[0], 'EMAIL')

  

    if data[1] == "URL":
      # print(data[0])
      check = is_URL(data[0])
      if len(check) > 0 and  check[0][1] - check[0][0] == len(data[0]):
        data = (data[0], 'URL')
      else: 
        data = (data[0], 'O')
      
    try:
      if data[1] != 'URL' and data[1] != 'EMAIL':
        check = is_URL(data[0])
        if len(check) > 0 and  check[0][1] - check[0][0] == len(data[0]):
          data = (data[0], 'URL')
    except:
      print(check)
    datas_trained.append(data)
  return datas_trained
    



from pyvi import ViTokenizer, ViPosTagger

def hau_xu_ly(sent, out):
  token, label = list(zip(*out))
  parts = sent.split()
  datas = []
  count = 0

  for i in range(len(parts)):
    word = parts[i]
    # print(word, count)
    try:
      
      # print(word, count)
      word_updated, index = gheptu(word,token , count)
    except:
    #  print(word_updated)
     print('error: {}, {}, {}'.format(word,token , count))
     break

    out_qdn = quet_dinh_nhan((word_updated, label[count: index]))
    
    for o in out_qdn:
      w, label_merged = o
      datas.append((w, label_merged))
    count = index


  datas_trained = preprocess_email_url(datas)
  # print(datas_trained)
  gomcum = gom_cum(datas_trained)
  # print(gomcum)
  if len(gomcum) != 0:
    indexs = cluster(gomcum, 2)
    # print(indexs)
    for index in indexs:
      string, label = list(zip(*datas_trained[index[0]: index[-1] + 1]))
      # string_loc = " ".join(string)
      if is_ADDRESS(string, label) == True:
        for i in range(index[0], index[-1] + 1):
          # print('hdhdhd')
          datas_trained[i] =(datas_trained[i][0], "ADDRESS")
  return datas_trained
# else:


def gom_cum(tokens):
  indexs = []
  for index in range(len(tokens)):
    token = tokens[index]
    if token[1] == "LOCATION" or token[1] == "ADDRESS" :
      indexs.append(index)
  return indexs

def cluster(data, maxgap):
    '''Arrange data into groups where successive elements
       differ by no more than *maxgap*

        >>> cluster([1, 6, 9, 100, 102, 105, 109, 134, 139], maxgap=10)
        [[1, 6, 9], [100, 102, 105, 109], [134, 139]]

        >>> cluster([1, 6, 9, 99, 100, 102, 105, 134, 139, 141], maxgap=10)
        [[1, 6, 9], [99, 100, 102, 105], [134, 139, 141]]

    '''
    data.sort()
    groups = [[data[0]]]
    for x in data[1:]:
        if abs(x - groups[-1][-1]) <= maxgap:
            groups[-1].append(x)
        else:
            groups.append([x])
    return groups
  

def gheptu(word, parts, index):
  
  if parts[index] == "[UNK]":
    return word, index + 1
    
  start = index
  # print(word)
  for i in range(start, len(parts) + 1):
    end = i
    # print("".join(parts[start:end]))
    # print(word.lower(),"".join(parts[start:end]).lower())



    if word.lower() == "".join(parts[start:end]).lower():
      # print('oke')
      # print(word, end)
      return word, end
  # print('fuck')
  # print(word, index)
  return None

def has_numbers(inputString):
  parts = inputString.split()
  # print(parts)
  for i in range(len(parts)):
    part = parts[i]
    for char in part:
      if char.isdigit():
        # print(i)
        if i > 0 and parts[i-1].lower() in ["quận", "q."]:
          return False
        else:
          return True
  return False

def is_ADDRESS(string, label):
  index_dau = [i for i, e in enumerate(string) if e == ","]
  index_not_dau_phay = [i for i, e in enumerate(label) if e == "O"]

  uy_tin = 0
  string_loc = " ".join(string)
  # print(label)
 
  # print(string)
  if 'ADDRESS' in label:
    uy_tin += 0.2
  
  # if '(' in string_loc or ')' in string_loc:
  #   uy_tin -= 0.025
  

  if has_numbers(string_loc):
    uy_tin += 0.05
  
  count =  len(index_dau) 

  # count = label.count('LOCATION')
  # print(count)
  if count > 0 and count < 3:  #count = 1, 2:
    uy_tin += 0.15
  elif count > 2:
    uy_tin += 0.2
  
  for i in index_not_dau_phay:
      if string[i] != ",":
        uy_tin -= 0.05
  level = ["số", "đường","tổ", "ngõ", "toà", "ngách", "hẻm","kiệt", "chung_cư", "ấp" ,"thôn", "khu","phố" , "quận", "phường", "xã", "thị_xã","huyện", "thành", "tp", "tỉnh" ]
  level_0 ={'status': True,'keywords': ["toà", "chung_cư", "số"] }
  level_1 = {'status': True, 'keywords': ["đường", "ngõ", "ngách", "hẻm","kiệt",]}
  level_2 = {'status': True, 'keywords':["ấp" ,"thôn", "khu","phố" , "quận", "phường", "xã", "tổ", "dân_phố"]}
  level_3 = {'status': True,'keywords':["thị","huyện"]}
  level_4 = {'status': True,'keywords':["thành", "tp", "tỉnh"]}

  parts =  ViPosTagger.postagging(ViTokenizer.tokenize(string_loc))[0]

  for seg_word in parts:
    # print(seg_word)
    if seg_word.lower() in level:
 

      if seg_word.lower() in level_0['keywords'] and level_0['status'] == True:
        uy_tin += 0.125
        level_0['status'] = False

      if seg_word.lower() in level_1['keywords'] and level_1['status'] == True:
        uy_tin += 0.125
        level_1['status'] = False

      elif seg_word.lower()  in level_2['keywords'] and level_2['status'] == True:
        uy_tin += 0.1
        level_2['status'] = False
      elif seg_word.lower() in  level_3['keywords'] and level_3['status'] == True:
   
        uy_tin += 0.05
        level_3['status'] = False
      elif seg_word.lower() in level_4['keywords'] and level_4['status'] == True:
     
        uy_tin += 0.025
        level_4['status'] = False

      
      # print(word.lower(), level_1.index(word.lower()) + 1)
      

  # print("check{}".format(uy_tin))
  if uy_tin >= 0.3:
    return True
  else:
    return False

###########################################################################################################################
import numpy as np

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()