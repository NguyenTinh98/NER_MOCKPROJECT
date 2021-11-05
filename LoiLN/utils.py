import pickle
from collections import defaultdict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from IPython.display import HTML as html_print
###################
def load_data_tags(path, name):
    """
    Load and return data

    Argument:
    --path: data path
    --name: .txt or .pkl

    Return:
    --data: format [(word, tag),...,(word, tag)]
    """
    if name == 'pkl':
        with open(path, 'rb') as f:
            data = pickle.load(f)
        return data
    with open(path, 'r') as f:
        s = f.readlines()
        data = []
        temp = []
        for i in s:
            a = i.split(sep=' ')
            if len(a)==2:
                temp.append((a[0],a[1].replace("\n","")))
            elif len(a)==1 and temp!=[]:
                data.append(temp)
                temp = []
    return data
#######################
def merge_tags(data):
    """
    Merge tags B-LOC, I-LOC to LOC example

    Argument:
    --data: output from  load_data_tags()

    Return:
    --data: new data with new format of tags
    """
    sequences = []
    for item in data:
        temp = []
        for i in item:
            lb = i[1].replace("B-","")
            lb = lb.replace("I-","")
            temp.append((i[0],lb))
        sequences.append(temp)
    return sequences
#######################
def cutting_sequences(sequences, max_len=256):
    """
    Split sequences which longer 256 words by '.'

    Argument:
    --sequences: data format [(word, tag),...,[word,tag]], maybe applying cutting if necessary

    Return:
    --data: data [(word, tag),...,[word,tag]] format with all of sentences have length <= 256
    """
    long_sequences = [seq for seq in sequences if len(seq) >= max_len]
    shorter_sequences = []
    for line in long_sequences:
        idx = 0
        for i, (word, tag) in enumerate(line):
            if word == '.':
                shorter_sequences.append(line[idx:i+1])
                idx = i+1
    data = [seq for seq in sequences if len(seq) < max_len]
    data += shorter_sequences
    return data
##########################
def word_to_sequences(data):
    """
    Transfer [(word, tag),...,[word,tag]] format to list words and list tags

    Argument:
    --data: [(word, tag),...,[word,tag]] format

    Return:
    --x: sentences
    --y: labels
    """
    X = []
    Y = []
    for line in data:
        x = [word for word, tag in line]
        y = [tag for word, tag in line]
        X.append(x.copy())
        x.clear()
        Y.append(y.copy())
        y.clear()
    return X, Y
######################
class transform_x():
    def __init__(self, max_len, vocabs):
        self.word2idx = defaultdict(lambda : 1) #OOV: 1, No word: 0
        self.max_len = max_len
        self.vocab_size = len(vocabs) + 2 # padding ; OOV
        self.words = vocabs

    def fit(self, X):
        """
        Emcoding X to training

        Argument:
        --X: sequence data format [[w1, w2..], [w1, w2,...]....]

        Return: 
        --X: sequence data format [[v1, v2..], [v1, v2,...]....] to can be train
        """

        for index, word in enumerate(self.words):
            self.word2idx[word] = index + 2
        X = [[self.word2idx.get(word, 1) for word in x] for x in X ]
        X = pad_sequences(X, maxlen = self.max_len, padding='post')
        return X

    def trans(self, X):
        X = [[self.word2idx.get(word, 1) for word in x] for x in X ]
        X = pad_sequences(X, maxlen = self.max_len, padding='post')
        return X
##########################
class transform_y():
    def __init__(self, max_len):
        self.max_seq_len = max_len
        self.tag_values  = sorted(['LOC', 'ORG', 'PER', 'MISC', 'O', 'PADDING'], key = lambda x: len(x))
        self.tag2idx = {tag : idx for idx,tag in enumerate(self.tag_values)}
        self.idx2tag = {idx  : tag for idx,tag in enumerate(self.tag_values)}
        
    def to_onehot(self, Y):
        """
        Convert value to one-hot value

        Argument:
        --Y: [1,2,3]
        Return:
        tags: [[0,1,0,0],
                [0,0,1,0],
                [0,0,0,1]]
        """
        labels = [[self.tag2idx[tag] for tag in line] for line in Y]
        padded_labels = pad_sequences(labels, maxlen=self.max_seq_len, padding='post', value=self.tag2idx['PADDING'])

        tags = [to_categorical(tags, num_classes=len(self.tag_values)) for tags in padded_labels]
        tags = np.array(tags)

        return tags
########################
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
        description: tags có thể là ['O','PER','LOC','ORG','MISC']
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
                lst.append(cstr(word, color='purple'))
            else:
                lst.append(cstr(word, color='yellow'))
    return print_color(lst)

def text_visualize():
    return print_color(['O',' -  ',cstr('PERSON','red'),
                        ' -  ',cstr('ORGANIZATION','blue'),
                        ' -  ',cstr('LOCATION','DarkGreen'),' -  ',cstr('MISC','Violet')])
######################