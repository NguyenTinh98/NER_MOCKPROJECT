


from keras.preprocessing.sequence import pad_sequences  
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

class BERT_DATALOADER:
    def __init__(self, dataset, tokenizer, tag_values,  device):
        self.dataset = dataset
        self.X , self.Y = split_data(self.dataset)
        self.MAX_LEN = 256
        self.BATCH_SIZE = 32
        self.Epoch = 60
        self.Patient = 15
        self.tag_values = ['PAD'] + tag_values 
        self.tag2idx = {t: i for i, t in enumerate(self.tag_values)}
        self.device = device
        self.tokenizer = tokenizer
    
    def create_dataloader(self, mode = 'evaluation', type = 'train'):
        X_subword, y_subword = self.add_subword2data()
        X_padding, y_padding, attention_masks = self._padding_data(X_subword,y_subword)
        X_tensor,y_tensor,masks = self._covert2tensor(X_padding, y_padding, attention_masks, mode)
        if type == 'train':
            train_data = TensorDataset(X_tensor, masks, y_tensor)
            train_sampler = RandomSampler(train_data)
            train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size = self.BATCH_SIZE)
            return train_dataloader
        elif type == 'dev' or type == 'test':
            valid_data = TensorDataset(X_tensor, masks, y_tensor)
            valid_sampler = SequentialSampler(valid_data)
            valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size = self.BATCH_SIZE)
            return valid_dataloader
    
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
    
    def _padding_data(self,X_subword,y_subword):
        '''
            input:
                X = [['Phạm', 'Văn', 'M', '##ạnh',..],....]
                Y = [['B-PER', 'I-PER','I-PER','I-PER',..],...]

            output: 
            [[10,20,30,40,0,0,0,0,0,0,0,0...],...],
            [[1, 2,3,4,5,5,5,5,5,5,5,5,5,...],...]
        '''
        X_padding = pad_sequences([self.tokenizer.convert_tokens_to_ids(txt) for txt in X_subword],
                          maxlen=self.MAX_LEN, dtype="long", value=0.0,
                          truncating="post", padding="post")

        y_padding = pad_sequences([[self.tag2idx.get(l) for l in lab] for lab in y_subword],
                        maxlen=self.MAX_LEN, value=self.tag2idx["PAD"], padding="post",
                        dtype="long", truncating="post")
        attention_masks = [[float(i != 0.0) for i in ii] for ii in X_padding]
        return X_padding, y_padding,attention_masks
    
    def _covert2tensor(self, X_padding, Y_padding, attention_masks, mode):
        if mode == 'training':
            X_tensor = torch.tensor(X_padding).to(self.device) 
            y_tensor = torch.tensor(Y_padding).to(self.device) 
            masks = torch.tensor(attention_masks).to(self.device)  

        elif mode =='evaluation':
            X_tensor = torch.tensor(X_padding).type(torch.LongTensor).to(self.device) 
            y_tensor = torch.tensor(Y_padding).type(torch.LongTensor).to(self.device) 
            masks = torch.tensor(attention_masks).type(torch.LongTensor).to(self.device) 
        return  X_tensor, y_tensor, masks


#########################################################################################################################
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
