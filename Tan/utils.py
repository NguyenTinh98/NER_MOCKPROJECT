from keras.preprocessing.sequence import pad_sequences
import torch 
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AutoTokenizer
import pandas as pd
import numpy as np

class process_data_for_BERT():
    def __init__(self,dir,tokenizer,tag_values,MAXLEN,type='BIO'):
        self.dir = dir
        self.tokenizer = tokenizer
        self.MAXLEN = MAXLEN

        tag_values.append("PAD")
        self.tag2idx = {t: i for i, t in enumerate(tag_values)}

        self.data = self.read_txt()
        if type == 'IO':
            self.data = self.BIO_to_IO_label(self.data)
        self.X, self.Y = self.split_data()
        
        self.X_subword, self.y_subword = self.add_subword2data()

        self.X_padding, self.Y_padding = self.padding_data()

        self.attention_masks = [[float(i != 0.0) for i in ii] for ii in self.X_padding]

    def read_txt(self):
        '''
            Reading data from txt to list (word - tag)
        '''
        _out = []
        with open(self.dir ,'r',encoding='utf-8') as f:
            _data = f.read()
        for sqs in _data.split('\n\n'):
            _temp = []
            for sq in sqs.split('\n'):
                ww, tt = sq.split('\t')
                _temp.append((ww, tt))
            _out.append(_temp)
        return _out

    def split_data(self):
        #(x,y)=> X= [x...] , Y= [y....]
        X, Y = [], []
        for sent in self.data:
            temp_x = []
            temp_y = []
            for word in sent:
                temp_x.append(word[0])
                temp_y.append(word[1])
            X.append(temp_x)
            Y.append(temp_y)
        return X, Y
    
    def _add_subword(self, sentence, text_labels):
        '''
        input:
            sentence = ['Phạm', 'Văn', 'Mạnh']
            text_labels = ['B-PER', 'I-PER','I-PER']

        output: 
            ['Phạm', 'Văn', 'M', '##ạnh'],
            ['B-PER', 'I-PER', 'I-PER', 'I-PER']
        '''
        tokenized_sentence = []
        labels = []
        for word, label in zip(sentence, text_labels):
            subwords = self.tokenizer.tokenize(word)
            tokenized_sentence.extend(subwords)
            
            labels.extend([label] * len(subwords))
        return tokenized_sentence, labels

    def  BIO_to_IO_label(self, data):
        '''
        input: [[('Tân','B-PER'),...],...]
        output: [[('Tân','PER'),...],...]
        '''
        OUT = []
        for item in data:
            temp = []
            for i in item:
                lb = i[1].replace("B-","")
                lb = lb.replace("I-","")
                temp.append((i[0],lb))
            OUT.append(temp)
        return OUT
    
    def add_subword2data(self):
        '''
            input:
                sentence = [['Phạm', 'Văn', 'Mạnh',..],....]
                text_labels = [['B-PER', 'I-PER','I-PER',..],...]

            output: 
                [['Phạm', 'Văn', 'M', '##ạnh',..],....],
                [['B-PER', 'I-PER','I-PER','I-PER',..],...]
        '''
        tokenized_texts_and_labels = [self._add_subword(sent, labs) for sent, labs in zip(self.X, self.Y)]

        tokenized_texts = [token_label_pair[0] for token_label_pair in tokenized_texts_and_labels]
        labels = [token_label_pair[1] for token_label_pair in tokenized_texts_and_labels]
        return tokenized_texts,labels
    
    def padding_data(self):
        '''
            input:
                X = [['Phạm', 'Văn', 'M', '##ạnh',..],....]
                Y = [['B-PER', 'I-PER','I-PER','I-PER',..],...]

            output: 
            [[10,20,30,40,0,0,0,0,0,0,0,0...],...],
            [[1, 2,3,4,5,5,5,5,5,5,5,5,5,...],...]
        '''
        X_padding = pad_sequences([self.tokenizer.convert_tokens_to_ids(txt) for txt in self.X_subword],
                          maxlen=self.MAXLEN, dtype="long", value=0.0,
                          truncating="post", padding="post")

        y_padding = pad_sequences([[self.tag2idx.get(l) for l in lab] for lab in self.y_subword],
                        maxlen=self.MAXLEN, value=self.tag2idx["PAD"], padding="post",
                        dtype="long", truncating="post")
        return X_padding, y_padding
    
    def covert2tensor(self, mode = 'training'):
        if mode == 'training':
            X_tensor = torch.tensor(self.X_padding).cuda()
            y_tensor = torch.tensor(self.Y_padding).cuda() 
            masks = torch.tensor(self.attention_masks).cuda() 

        elif mode =='evaluation':
            X_tensor = torch.tensor(self.X_padding).type(torch.LongTensor).cuda() 
            y_tensor = torch.tensor(self.Y_padding).type(torch.LongTensor).cuda() 
            masks = torch.tensor(self.attention_masks).type(torch.LongTensor).cuda() 
        return  X_tensor, y_tensor, masks

    def Dataloader(self, BATCH_SIZE, mode = 'training' ,type = 'train'):
        X_tensor,y_tensor,masks = self.covert2tensor(mode)
        if type == 'train':
            train_data = TensorDataset(X_tensor, masks, y_tensor)
            train_sampler = RandomSampler(train_data)
            train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=BATCH_SIZE)
            return train_dataloader
        elif type == 'valid':
            valid_data = TensorDataset(X_tensor, masks, y_tensor)
            valid_sampler = SequentialSampler(valid_data)
            valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=BATCH_SIZE)
            return valid_dataloader

def check_label(data):
    '''
    input: [[('Hello','O'),...],...]
    output: {'O','LOC',"ORG",...}
    '''
    a = []
    for i in data:
        for j in i:
            _, l = j
            a.append(l)
    return set(a)

def get_count_tags(data):
    label = dict()
    for x in data:
        for _,t in x:
            if t not in label.keys():
                label[t] = 1
            else:
                label[t] +=1
    return label

def DF_LABEL(data_train, data_valid):
    tag_train = get_count_tags(data_train)
    tag_val = get_count_tags(data_valid)
    df = pd.DataFrame(list([tag_train, tag_val]))
    df.set_index([pd.Index(['train_set','valid_set'])])
    return df

