from spacy import displacy

COLORS ={
    'EMAIL':'gray', 
    'ADDRESS':'maroon',
    'PERSON':'red',
    'PHONENUMBER': 'purple',
    'MISCELLANEOUS':'fuchsia',
    'QUANTITY':'green',
    'PERSONTYPE':'lime',
    'ORGANIZATION':'olive',
    'PRODUCT':'yellow',
    'SKILL':'navy',
    'IP':'blue',
    'LOCATION':'teal',
    'DATETIME':'aqua',
    'EVENT':'darkorange',
    'URL':'deeppink'
}
NER = list(COLORS.keys())

OPTIONS = {'ents': NER, 'colors': COLORS}
    
## visualize result
## input: predict format [(word, tag)]

def BERT_VISUALIZE(arr):
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
    return displacy.render(ex, manual=True, jupyter=True, style='ent', options = OPTIONS )