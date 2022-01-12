from flask import Flask,url_for,render_template,request
import spacy
from spacy import displacy
import json
import torch
import time
from main import *
import utils

def loading(PATH = 'model/xlmr_span1.pt'):
    MAX_LEN = 256
    BS = 64
    DROPOUT_OUT = 0.4           
    model_name = 'xlmr_softmax'
    device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
    print(device)
    tag_values = ['PAD','ADDRESS','SKILL','EMAIL','PERSON','PHONENUMBER','MISCELLANEOUS','QUANTITY','PERSONTYPE',
            'ORGANIZATION','PRODUCT','IP','LOCATION','O','DATETIME','EVENT', 'URL']  
    with open('path.json', 'r', encoding= 'utf-8') as f:
        dict_path = json.load(f)
    dict_path = dict_path[model_name.split('_')[0]]
    dict_path['weight'] = PATH
    start = time.time()
    print("1. Loading some package")
    ner = NER(dict_path = dict_path, model_name = model_name, tag_value = tag_values , dropout = DROPOUT_OUT, max_len = MAX_LEN, batch_size = BS, device = device, concat=False)
    print(f"===== Done !!! =====Time: {time.time() -start:.4} s =========")
    print('2.Load model')
    start = time.time()
    ner.model = load_model(ner.model,dict_path['weight'],device)
    print(f"===== Done !!! =====Time: {time.time() -start:.4} s =========")
    return ner

ner = loading('model/span2_pool_nocat/xlmr_span2_10t01_pool_nocat.pt')

HTML_WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem">{}</div>"""

from flaskext.markdown import Markdown

app = Flask(__name__)
Markdown(app)

@app.route('/')
def index():
	return render_template('result.html', raw_text = '', result='', result1='')


@app.route('/',methods=["GET","POST"])
def extract():
    if request.method == 'POST':
        if request.form.get('submit_button', False) == "NE Recognition":
            raw_text = request.form['rawtext']
            docx, outx = ner.predict(raw_text)

            html = utils.visualize_spacy(docx)
            html = html.replace('[/n]','</br>')
            result = HTML_WRAPPER.format(html)

            html1 = utils.visualize_spacy(outx)
            html1 = html1.replace('[/n]','</br>')
            result1 = HTML_WRAPPER.format(html1)
        elif request.form.get('clear_button', False) == "Clear":
            raw_text = ''
            result = ''
            result1= ''
    
    return render_template('result.html', raw_text=raw_text, result=result, result1=result1)

if __name__ == '__main__':
    app.run(port=5000, debug=True, use_reloader=False)
