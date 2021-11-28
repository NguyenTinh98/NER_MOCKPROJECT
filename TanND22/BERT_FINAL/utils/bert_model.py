from TorchCRF import CRF
import torch.nn as nn
import torch
import torch.nn.functional as F
log_soft = F.log_softmax

class BERT_4_CRF(nn.Module):
    def __init__(self, bert_model, num_labels):
        super(BERT_4_CRF, self).__init__()
        self.bert = bert_model
        self.dropout = nn.Dropout(0.25)
        # 4 last of layer
        self.classifier = nn.Linear(4*768, num_labels)
        self.crf = CRF(num_labels, batch_first = True)
    
    def forward_custom(self, b_input_ids, b_input_mask,  b_labels=None, token_type_ids=None):
        outputs = self.bert(b_input_ids, attention_mask=b_input_mask)
        sequence_output = torch.cat((outputs[1][-1], outputs[1][-2], outputs[1][-3], outputs[1][-4]),-1)
        sequence_output = self.dropout(sequence_output)
        
        emission = self.classifier(sequence_output) # [32,256,17]
        
        if b_labels is not None:
            loss = -self.crf(log_soft(emission, 2), b_labels, mask=b_input_mask.type(torch.uint8), reduction='mean')
            prediction = self.crf.decode(emission, mask=b_input_mask.type(torch.uint8))
            return [loss, prediction]
                
        else:
            prediction = self.crf.decode(emission, mask=b_input_mask.type(torch.uint8))
            return prediction



class BERT_4_SOFTMAX(nn.Module):
    def __init__(self, bert_model, num_labels):
        super(BERT_4_SOFTMAX, self).__init__()
        self.num_labels = num_labels
        self.bert = bert_model
        self.dropout = nn.Dropout(0.25)
        # 4 last of layer
        self.classifier = nn.Linear(4*768, num_labels)

    def forward_custom(self, input_ids, attention_mask=None, labels=None, head_mask=None):
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




######################################################################################################################
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    '''Multi-class Focal loss implementation'''
    def __init__(self, gamma=2, weight=None,ignore_index=-100):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.ignore_index=ignore_index

    def forward(self, input, target):
        """
        input: [N, C]
        target: [N, ]
        """
        logpt = F.log_softmax(input, dim=1)
        pt = torch.exp(logpt)
        logpt = (1-pt)**self.gamma * logpt
        loss = F.nll_loss(logpt, target, self.weight,ignore_index=self.ignore_index)
        return loss


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, eps=0.1, reduction='mean',ignore_index=-100):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.eps = eps
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, output, target):
        c = output.size()[-1]
        log_preds = F.log_softmax(output, dim=-1)
        if self.reduction=='sum':
            loss = -log_preds.sum()
        else:
            loss = -log_preds.sum(dim=-1)
            if self.reduction=='mean':
                loss = loss.mean()
        return loss*self.eps/c + (1-self.eps) * F.nll_loss(log_preds, target, reduction=self.reduction,
                                                           ignore_index=self.ignore_index)

################################################################################################################
# Define bert 4 layer
class BERT_LSTM_SOFTMAX(nn.Module):
    def __init__(self, bert_model, num_labels):
        super(BERT_LSTM_SOFTMAX, self).__init__()
        self.num_labels = num_labels
        self.bert = bert_model
        self.classifier_1 = nn.Linear(768, num_labels)
        self.dropout_1 = nn.Dropout(0.2)
        self.classifier_2 = nn.Linear(768, num_labels)
        self.dropout_2 = nn.Dropout(0.2)
        self.classifier_3 = nn.Linear(768, num_labels)
        self.dropout_3 = nn.Dropout(0.2)
        self.classifier_4 = nn.Linear(768, num_labels)
        self.dropout_4 = nn.Dropout(0.2)
        self.classifier = nn.Linear(256 + 768, num_labels)
        self.dropout = nn.Dropout(0.25)
        
        self.lstm = nn.LSTM(input_size = 4*num_labels ,hidden_size = 256 , num_layers = 4*num_labels, dropout = 0.2)
        

        
        
    def forward_custom(self, input_ids, attention_mask=None, 
                       head_mask=None, labels=None):
        outputs = self.bert(input_ids = input_ids, attention_mask=attention_mask)
        out_bert = outputs[1][-1]
        out_1 = self.dropout_1(outputs[1][-1])
        out_2 = self.dropout_1(outputs[1][-2])
        out_3 = self.dropout_1(outputs[1][-3])
        out_4 = self.dropout_1(outputs[1][-4])
        sequence_1 = self.classifier_1(out_1)
        sequence_2 = self.classifier_1(out_2)
        sequence_3 = self.classifier_1(out_3)
        sequence_4 = self.classifier_1(out_4)
        
        sequence_output = torch.cat((sequence_1, sequence_2, sequence_3, sequence_4),-1)
        lstm_output = self.lstm(sequence_output)
        sequence_output = torch.cat((lstm_output[0], out_bert),-1)
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output) # bsz, seq_len, num_labels
        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        
        if labels is not None:
            loss_fct = FocalLoss(ignore_index=0)
            if attention_mask is not None:
                    active_loss = attention_mask.view(-1) == 1
                    active_logits = logits.view(-1, self.num_labels)[active_loss]
                    active_labels = labels.view(-1)[active_loss]
                    loss = loss_fct(active_logits, active_labels)
            else:
                    loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs
        return outputs  