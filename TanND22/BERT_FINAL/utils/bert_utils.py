import pickle
def read_dataset(dir_train):
    with open(dir_train ,'rb') as f:
        _data = pickle.load(f)
    data = [sq for sq in _data if len(sq) >= 0]
    return data

def split_data(data):
    #(x,y)=> X= [x...] , Y= [y....]
    X, Y = [], []
    for sent in data:
        temp_x = []
        temp_y = []
        for word in sent:
            temp_x.append(word[0])
            temp_y.append(word[1])
        X.append(temp_x)
        Y.append(temp_y)
    return X, Y

