import torch
import torch.nn as nn

class BERT_SOFTMAX(nn.Module):
    def __init__(self, bert_model, num_labels):
        super(BERT_SOFTMAX, self).__init__()
        self.num_labels = num_labels
        self.bert = bert_model
        self.dropout = nn.Dropout(0.25)
        # 4 last of layer
        self.classifier = nn.Linear(4*768, num_labels)

    
    def forward_custom(self, input_ids, attention_mask=None, 
                       head_mask=None, labels=None):
        outputs = self.bert(input_ids = input_ids, attention_mask=attention_mask)
        sequence_output = torch.cat((outputs[1][-1], outputs[1][-2], outputs[1][-3], outputs[1][-4]),-1)
        sequence_output = self.dropout(sequence_output)
        
        logits = self.classifier(sequence_output) # bsz, seq_len, num_labels
        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=0)
            if attention_mask is not None:
                    active_loss = attention_mask.view(-1) == 1
                    active_logits = logits.view(-1, self.num_labels)[active_loss]
                    active_labels = labels.view(-1)[active_loss]
                    loss = loss_fct(active_logits, active_labels)
            else:
                    loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs
        return outputs  #scores, (hidden_states), (attentions)    




import torch
from pyvi import ViTokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np

def transform_test(test="", mode='token'):
  # ['bộ', 'văn_hóa', 'và', 'truyền_thông']
  tokens = ViTokenizer.tokenize(test).split()
  temp = []
  for i, w in enumerate(tokens):
    if mode == 'token':
        k = w.replace("_", " ")
        temp.append(k)
    else:   # mode == 'word'
        k = w.replace("_", " ").split()
        for j in k:
            temp.append(j)
  return temp

def tokenize_predict(tokenizer, sentence):
  '''
    sentence: ['văn_hóa','và','nghệ_thuật']
    
    output: ['văn_@@', 'h@@', 'ó@@', 'a', 'và', 'nghệ_thuật']
  '''
  subwords = []

  for word in sentence:
    subword = tokenizer.tokenize(word)

    subwords.extend(subword)

  return subwords

def predict_text(model, tokenizer, tag_values, test_sentence, device):
    #predict with model
    model.eval()
    test_sentence_token = transform_test(test_sentence, 'word')
    subwords = tokenize_predict(tokenizer, test_sentence_token)
    input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(subwords)],
                              maxlen=512, dtype="long", value=0.0,
                              truncating="post", padding="post")
    input_ids_tensor = torch.tensor(input_ids).type(torch.LongTensor).to(device) #Fixfug here
    input_mask = [[float(i != 0.0) for i in ii] for ii in input_ids]
    input_mask_tensor = torch.tensor(input_mask).type(torch.LongTensor).to(device)
    with torch.no_grad():
        #outputs = model(input_ids_tensor, token_type_ids = None, attention_mask = input_mask_tensor)
        outputs = model.forward_custom(input_ids=input_ids_tensor, attention_mask=input_mask_tensor)
    logits = outputs[0].detach().cpu().numpy()
    
    #Precroces subword
    
    len_subword = sum(input_ids[0] != 0)
    tokens = tokenizer.convert_ids_to_tokens(input_ids_tensor[0].to('cpu').numpy())[:len_subword]
    predict = np.argmax(logits, axis=2)[0][:len_subword]
    
    tags_predict = [ tag_values[i]  for i in  predict]
    
    tags = []
    tests = []
    for index in range(len(tokens)):
        if "##" not in tokens[index]:
            tags.append(tags_predict[index])
            tests.append(tokens[index])
        else:
            tests[-1] = tests[-1] + tokens[index].replace("##","")
    
    return [(w,t) for w,t in zip(tests,tags)]





######################################################################################################################
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
    X = [[w for w,_ in sq] for  sq in dataset][idx]
    Y = [[t for _,t in sq] for  sq in dataset][idx]
    X_Sub, Y_Sub = add_subword(tokenizer, X, Y)
    X_padding, Y_padding = padding_data(tokenizer,X_Sub,Y_Sub,MAXLEN,tag2idx)
    input_ids_tensor = torch.tensor(X_padding).type(torch.LongTensor).to(device)
    input_mask = [[float(i != 0.0) for i in ii] for ii in X_padding]
    input_mask_tensor = torch.tensor(input_mask).type(torch.LongTensor).to(device)
    with torch.no_grad():
        outputs = model.forward_custom(input_ids_tensor, attention_mask = input_mask_tensor)
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