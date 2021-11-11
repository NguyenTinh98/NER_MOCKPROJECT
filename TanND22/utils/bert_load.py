from keras.preprocessing.sequence import pad_sequences
import torch 
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AutoTokenizer
import pandas as pd
import numpy as np

class process_data_for_BERT():
    def __init__(self,dir,tokenizer,tag_values,MAXLEN,type='BIO',device = 'cpu'):
        self.dir = dir
        self.tokenizer = tokenizer
        self.MAXLEN = MAXLEN
        self.device = device

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
            X_tensor = torch.tensor(self.X_padding).to(self.device)
            y_tensor = torch.tensor(self.Y_padding).to(self.device)
            masks = torch.tensor(self.attention_masks).to(self.device)

        elif mode =='evaluation':
            X_tensor = torch.tensor(self.X_padding).type(torch.LongTensor).to(self.device)
            y_tensor = torch.tensor(self.Y_padding).type(torch.LongTensor).to(self.device)
            masks = torch.tensor(self.attention_masks).type(torch.LongTensor).to(self.device)
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




################################################################
def add_subword(tokenizer, X,  Y):
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
        for word, label in zip(X, Y):
            subwords = tokenizer.tokenize(word)
            tokenized_sentence.extend(subwords)
            
            labels.extend([label] * len(subwords))
        return tokenized_sentence, labels

def padding_data(tokenizer,X_subword,y_subword,MAXLEN,tag2idx):
        '''
            input:
                X = [['Phạm', 'Văn', 'M', '##ạnh',..],....]
                Y = [['B-PER', 'I-PER','I-PER','I-PER',..],...]

            output: 
            [[10,20,30,40,0,0,0,0,0,0,0,0...],...],
            [[1, 2,3,4,5,5,5,5,5,5,5,5,5,...],...]
        '''
        X_padding = pad_sequences([tokenizer.convert_tokens_to_ids(X_subword)],
                          maxlen=MAXLEN, dtype="long", value=0.0,
                          truncating="post", padding="post")

        y_padding = pad_sequences([[tag2idx.get(l) for l in y_subword]],
                        maxlen=MAXLEN, value=tag2idx["PAD"], padding="post",
                        dtype="long", truncating="post")
        
        return X_padding,y_padding

def predict_data_train(model, tokenizer, tag_values, dataset , idx, tag2idx, MAXLEN, device):
    """
        input:
            - model:
            - tokenizer:
            - tag_value: ['O', 'PER', 'LOC',...]
            - dataset: [[('anh','O'),...]]
            - idx: index in dataset
            - tag2idx: {'PER': 0, 'LOC': 1, 'ORG': 2, 'MISC': 3, 'O': 4, 'PAD': 5}
            - MAXLEN: 256,512,...
    """
    X = dataset.X[idx]
    Y = dataset.Y[idx]
    X_Sub, Y_Sub = add_subword(tokenizer, X, Y)
    X_padding, Y_padding = padding_data(tokenizer,X_Sub,Y_Sub,MAXLEN,tag2idx)
    input_ids_tensor = torch.tensor(X_padding).type(torch.LongTensor).to(device)
    input_mask = [[float(i != 0.0) for i in ii] for ii in X_padding]
    input_mask_tensor = torch.tensor(input_mask).type(torch.LongTensor).to(device)
    with torch.no_grad():
        outputs = model(input_ids_tensor, token_type_ids = None, attention_mask = input_mask_tensor)

    logits = outputs[0].detach().cpu().numpy()

        #Precroces subword

    len_subword = sum(X_padding[0] != 0)
    tokens = tokenizer.convert_ids_to_tokens(input_ids_tensor[0].to('cpu').numpy())[:len_subword]
    predict = np.argmax(logits, axis=2)[0][:len_subword]

    tags_predict = [tag_values[i]  for i in  predict]
    tags_true = [tag_values[i]  for i in  Y_padding[0]]
    y_predict = []
    words = []
    y_true = []
    for index in range(len(tokens)):
        if "##" not in tokens[index]:
            y_predict.append(tags_predict[index])
            y_true.append(tags_true[index])
            words.append(tokens[index])
        else:
            words[-1] = words[-1] + tokens[index].replace("##","")
    return [(w,'?',t) for w,t in zip(words,y_true)], y_predict