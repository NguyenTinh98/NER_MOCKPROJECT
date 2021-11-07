from IPython.display import HTML as html_print

def cstr(s,color='black'):
    '''
        input: 'Tân'
        output: 'Tag HTML of word'
    '''
    return "<text style='color:{};font-size:150%'><b> {} </b></text>".format(color,s)

def print_color(lst):
    '''
        input: list of word, ex:['Chào', 'Tân', 'xinh', 'đẹp']
        output: show color of text
    '''
    return html_print(' '.join(lst))

def visualize(predict_):
    '''
        description: định dạng đầu vào là ['O','PER','LOC','ORG','MISC']
                    hoặc ['O','B-PER','I-PER','B-LOC','I-LOC','B-MISC','I-MISC']  
        input: [('Chào','O'),('Tân','PER'),('xinh','O'),('đẹp','O')]
        output: show colr of text
    '''
    lst =[]
    for predict in predict_:
        word,tag = predict
        if len(tag) == 1:
            lst.append(word)
        else:
            if tag == 'PER' or tag[2:] == 'PER':
                lst.append(cstr(word, color='red'))
            elif tag== 'ORG' or tag[2:] == 'ORG':
                lst.append(cstr(word, color='blue'))
            elif tag == 'LOC'or tag[2:] == 'LOC':
                lst.append(cstr(word, color='DarkGreen'))
            elif tag == 'MISC' or tag[2:] == 'MISC':
                lst.append(cstr(word, color='Violet'))
            else:
                lst.append(cstr(word, color='yellow'))
    return print_color(lst)
def test_visualize():
    return print_color(['O',' -  ',cstr('PERSON','red'),' -  ',cstr('ORGANIZATION','blue'),' -  ',cstr('LOCATION','DarkGreen'),' -  ',cstr('MISC','Violet')])


# predict =[('Tân',"B-PER"),('chào','O'),('mọi','O'),('người','O')]
# visualize(predict)
# text_visualize()

import time
import torch
from utils.bert_predict import predict_text

def BERT_PREDICT(PATH_MODEL, model, tokenizer, Tag, text, device, check_time = False):
    '''
        input:
            - Path_model: 'IO_BERT_MULTI.pt'
            - model: define and init model
            - Tag: ['PER','LOC','ORG','MISC','O']
            - Text: ' Tan dep trai'
            - device: 'cpu' or 'cuda'
            - check_time: True or False
    '''
    start = time.time()
    model.load_state_dict(torch.load(PATH_MODEL), strict=False)
    model.to(device)
    visual = visualize(predict_text(model, tokenizer, Tag ,text,device))
    if check_time == True:
        print("Time : ",time.time()-start," s")
    return visual