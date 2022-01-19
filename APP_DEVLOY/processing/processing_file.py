########################################################################################################################################
import pickle
import json

############################################################# PKL FILE #################################################################
def write_pickle(dt, path):
  with open(path, 'wb') as f:
    pickle.dump(dt, f)

def read_pickle(file):
  with open(file, 'rb') as f:
    return pickle.load(f)

############################################################# JSON FILE #################################################################
def write_json(dt, path):
  with open(path, 'w', encoding='utf-8') as f:
    json.dump(dt, f)

def read_json(path):
  with open(path, 'r', encoding='utf-8') as f:
    return json.load(f)

############################################################# TXT FILE #################################################################
def convert_pkl2txt(data,path):
    with open(path, 'w', encoding='utf-8') as f:
        for sq in data:
            for txt in sq:
                ww, tt = txt
                f.write(ww+'\t'+tt)
                f.write('\n')
            f.write('\n')

############################################################# CONVERT TAG IO TO BIO #################################################################
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

