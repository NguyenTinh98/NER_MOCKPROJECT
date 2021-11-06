from utils import *
import pickle
import pandas as pd

def evaluate(model, text, vocabs, max_len=256):
    """
    Testing real text
    Argument:
    --model: best model trained
    --text: real text
    Return:
    --data: [word, tag] format
    """
    text_tokens = [text.split()]
    trans_x = transform_x(max_len, vocabs)
    tokens = trans_x.fit(text_tokens)
    y_testing = model.predict(tokens)
    y_testing = np.argmax(y_testing[0], axis=-1).flatten()
    
    list_test = []
    trans_y = transform_y(max_len)
    for i in range(len(text_tokens[0])):
        list_test.append((text_tokens[0][i], 
                                trans_y.idx2tag[y_testing[i]]))
    return list_test