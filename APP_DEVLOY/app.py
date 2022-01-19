from flask import Flask,url_for,render_template,request, jsonify
import spacy
from spacy import displacy
import json
import torch
import time
from main import *
import utils
from tqdm import tqdm, trange
from sklearn.model_selection import train_test_split
from processing.pre_processing import preprocessing_text

from xai import xai_lime

import numpy as np

tag_values = ['PAD','ADDRESS','SKILL','EMAIL','PERSON','PHONENUMBER','MISCELLANEOUS','QUANTITY','PERSONTYPE',
            'ORGANIZATION','PRODUCT','IP','LOCATION','O','DATETIME','EVENT', 'URL']  
            
def loading(PATH = 'model/xlmr_span2_10t01_pool_nocat.pt'):
    MAX_LEN = 256
    BS = 64
    DROPOUT_OUT = 0.4           
    model_name = 'xlmr_softmax'
    device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
    print(device)
    
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

ner = loading('model/xlmr_span2_10t01_pool_nocat.pt')


from flaskext.markdown import Markdown

app = Flask(__name__,   static_url_path='', 
            static_folder='static')
Markdown(app)


@app.route('/interpret')
def interpret():
	return render_template('interpret.html')

@app.route('/predict')
def predict():
	return render_template('predict.html')


@app.route('/predict', methods =["POST"])
def api_predict():
    text = request.json['text']
    rs = ner.predict(text)
    out = {'rs': rs}
    return jsonify(out)

        

@app.route('/')
def index():
	return render_template('interpret.html', raw_text = '', result='')





@app.route('/interpret',methods=["POST"])
def api_interpret():
    # try:
    print("Server received data: {}, {}".format(request.json['text'], request.json['idx']))
    print("Server received data: {}, {}".format(type(request.json['text']), type(request.json['idx'])))
    text = request.json['text']
    id_word = int(request.json['idx'])

    rs = xai_lime(ner, text, id_word, tag_values)
    return jsonify(rs)
    # except:
    #     return jsonify({
    #         "message": "error in server"})



if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)








