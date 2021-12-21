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
