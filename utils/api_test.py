from requests import *
import json
import time

from spacy import displacy

COLORS ={
    'EMAIL':'orange',
    'ADDRESS':'olive',
    'PERSON':'red',
    'PHONENUMBER': 'gray',
    'MISCELLANEOUS':'fuchsia',
    'QUANTITY':'green',
    'PERSONTYPE':'pink',
    'ORGANIZATION':'yellow',
    'PRODUCT':'teal',
    'SKILL':'lime',
    'IP':'blue',
    'LOCATION':'purple',
    'DATETIME':'aqua',
    'EVENT':'darkorange',
    'URL':'brown'
}

NER = list(COLORS.keys())

OPTIONS = {'ents': NER, 'colors': COLORS}
    

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

while(True):
    text = input('Enter sentence: ')
    r = post(url='http://2816-34-66-29-55.ngrok.io/ocr/' ,
             data={'id': uuid.uuid4(), 'data':text})
             
    aa = json.loads(r.text)

    bertvisualize(aa['text'])
    
 ####
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

def preprocessing_text2(text):
    dictt = {'™': ' ', '‘': "'", '®': ' ', '×': ' ', '😀': ' ', '‑': ' - ', '́': ' ', '—': ' - ', '̣': ' ', '–': ' - ', '`': "'",\
             '“': '"', '̉': ' ','’': "'", '̃': ' ', '\u200b': ' ', '̀': ' ', '”': '"', '…': '...', '\ufeff': ' ', '″': '"'}
    text = text.split('\n')
    text = ' '.join([i.strip()  for i in text if i!=''])
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

def convertfile2doccano(path1, path2):
  res = []
  data = read_txt(path1)
  for line in data:
    res.append(preprocessing_text2(line.strip()))
  write_txt(res, path2)

