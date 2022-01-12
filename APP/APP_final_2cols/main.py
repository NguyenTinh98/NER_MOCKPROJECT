'''
train lại với tên biến 
    self.model
'''
from tqdm import tqdm
import torch.nn as nn
import torch
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from transformers import AutoConfig, AutoModel, AutoTokenizer
from sklearn.metrics import f1_score
from TorchCRF import CRF   
#from torchcrf import CRF
import torch.nn.functional as F
log_soft = F.log_softmax
import utils
import data_processing

class NER(nn.Module):
    def __init__(self, dict_path, model_name, tag_value ,dropout = 0.4,  max_len = 256, batch_size = 32, device = 'cuda', concat=True):
        super(NER, self).__init__()
        self.tag_value = tag_value
        self.max_len = max_len
        self.batch_size = batch_size
        self.device = device
        self.model_name = model_name
        self.is_crf = 'crf' in self.model_name
        self.tokenizer = AutoTokenizer.from_pretrained(dict_path['tokenizer'], do_lower_case=False,use_fast=False)
        self.config = AutoConfig.from_pretrained(dict_path['config'], output_hidden_states=True)
        self.base_model = AutoModel.from_pretrained(dict_path['model'], config=self.config, add_pooling_layer=True)
        self.concat = concat
        self.sub = dict_path['sub']
        self.dropout = dropout
        if self.is_crf:
            self.model = BaseBertCrf(self.base_model, self.dropout, num_labels= len(self.tag_value), concat = self.concat)
        else:
            self.model = BaseBertSoftmax(self.base_model, self.dropout, num_labels= len(self.tag_value), concat = self.concat)
        self.model.to(self.device)
           

    def predict(self, texts):
        texts = utils.preprocessing_text(self.tokenizer,texts)
        subwords = self.tokenizer.tokenize(texts)
        sub_cut = utils.cutting_subword(subwords, sub = self.sub, size = self.max_len)
        tags_out = []
        words_out = []
        probs_out = []
        self.model.eval()
        for sub in sub_cut:
            input_ids = pad_sequences([self.tokenizer.convert_tokens_to_ids(sub)],
                                        maxlen=self.max_len, dtype="long", value=self.tokenizer.pad_token_id,
                                        truncating="post", padding="post")
            input_ids_tensor = torch.tensor(input_ids).type(torch.LongTensor).to(self.device)
            input_mask = [[float(i != 1.0) for i in ii] for ii in input_ids]
            input_mask_tensor = torch.tensor(input_mask).type(torch.LongTensor).to(self.device) 
            with torch.no_grad():
                outputs = self.model.forward_custom(input_ids_tensor, input_mask_tensor)
            if self.is_crf:
                predict = outputs[0]
            else:
                logits = outputs[0].detach().cpu().numpy()
                len_subword = sum(input_ids[0] != 1)
                predict = np.argmax(logits, axis=2)[0][:len_subword]
                sm = [(utils.softmax(logits[0,i])) for i in range(len_subword)]

            tags_predict = [ self.tag_value[i]  for i in  predict]
            tests, tags, probs = utils.merge_subtags_4column(sub, tags_predict, sm)
            words_out += tests
            tags_out += tags
            probs_out += probs
        out1 = [(w,t,p) if w != '.</s>' else  (w.replace('.</s>','[/n]'),t) for w,t,p in zip(words_out,tags_out, probs_out)][1:-1]
        out = data_processing.span_cluster(out1)
        texts = " ".join([word for (word, _) in out])
        result = data_processing.post_processing(texts, out)
        return result
         
#################################################################################################################################################
class BaseBertCrf(nn.Module):
    def __init__(self, bert_model, drop_out, num_labels, concat = True):
        super(BaseBertCrf, self).__init__()
        self.concat = concat
        self.model = bert_model
        self.dropout = nn.Dropout(drop_out)
        if self.concat :
            self.classifier = nn.Linear(4*768, num_labels) # 4 last of layer
        else:
            self.classifier = nn.Linear(768, num_labels)
        self.crf = CRF(num_labels, batch_first = True)
        
    def forward_custom(self, b_input_ids, b_input_mask,  b_labels=None, token_type_ids=None):
        outputs = self.model(b_input_ids, attention_mask=b_input_mask)
        if self.concat:
            sequence_output = torch.cat((outputs[1][-1], outputs[1][-2], outputs[1][-3], outputs[1][-4]),-1)
            sequence_output = self.dropout(sequence_output)
        else:
            sequence_output = self.dropout(outputs[0])
        
        emission = self.classifier(sequence_output) # [32,256,17]
        if b_labels is not None:
            loss = -self.crf(log_soft(emission, 2), b_labels, mask=b_input_mask.type(torch.bool), reduction='mean')
            prediction = self.crf.decode(emission, mask=b_input_mask.type(torch.bool))
            return [loss, prediction]
                
        else:
            prediction = self.crf.decode(emission, mask=b_input_mask.type(torch.bool))
            return prediction

#################################################################################################################################################
class BaseBertSoftmax(nn.Module):
    def __init__(self, bert_model, drop_out ,num_labels , concat = True):
        super(BaseBertSoftmax, self).__init__()
        self.concat = concat
        self.num_labels = num_labels
        self.model = bert_model
        self.dropout = nn.Dropout(drop_out)
        if self.concat:
            self.classifier = nn.Linear(4*768, num_labels) # 4 last of layer
        else:
            self.classifier = nn.Linear(768, num_labels)
    
    def forward_custom(self, input_ids, attention_mask=None,
                        labels=None, head_mask=None):
        outputs = self.model(input_ids = input_ids, attention_mask=attention_mask)
        if self.concat:
            sequence_output = torch.cat((outputs[1][-1], outputs[1][-2], outputs[1][-3], outputs[1][-4]),-1)
            sequence_output = self.dropout(sequence_output)
        else:
            sequence_output = self.dropout(outputs[0])
        
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

#########################################################
def load_model(base_model, path, device):
    base_model.load_state_dict(torch.load(path, map_location=torch.device('cpu')), strict=False)
    base_model.to(device)
    return base_model

