
import torch
from tqdm import tqdm
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
import numpy as np


def valuation_bert_multi(model, valid_dataloader,tag_values,device):
    '''
        input: 
            - model
            - valid_dataloader
            - tag_values: ['O', 'PER', .... ]
            - device: cuda
        output:
            - report F1
            - loss 

    '''
    # ========================================
    #               Validation
    # ========================================
    # After the completion of each training epoch, measure our performance on
    # our validation set.

    # Put the model into evaluation mode
    model.eval()
    # Reset the validation loss for this epoch.
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    predictions , true_labels = [], []
    for batch in tqdm(valid_dataloader, desc = 'Progress Bar'):
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch

        # Telling the model not to compute or store gradients,
        # saving memory and speeding up validation
        with torch.no_grad():
            # Forward pass, calculate logit predictions.
            # This will return the logits rather than the loss because we have not provided labels.
            outputs = model(b_input_ids, token_type_ids=None,
                            attention_mask=b_input_mask, labels=b_labels)
        # Move logits and labels to CPU
        
        logits = outputs[1].detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        # Calculate the accuracy for this batch of test sentences.
        eval_loss += outputs[0].mean().item()
        predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
        true_labels.extend(label_ids)

    eval_loss = eval_loss / len(valid_dataloader)
    print("Validation loss: {}".format(eval_loss))
    pred_tags = [tag_values[p_i] for p, l in zip(predictions, true_labels)
                                 for p_i, l_i in zip(p, l) if tag_values[l_i] != "PAD"]
    valid_tags = [tag_values[l_i] for l in true_labels
                                  for l_i in l if tag_values[l_i] != "PAD"]
    print("Validation F1-Score: {}".format(f1_score(pred_tags, valid_tags,average='micro')))
    print(classification_report(valid_tags, pred_tags,digits = 4))
    print("####################")

#################################################################################################################

def valuation_bert_4_crf(model, tokenizer, valid_dataloader,tag_values,device,mode):
    '''
        input: 
            - model
            - valid_dataloader
            - tag_values: ['O', 'PER', .... ]
            - device: cuda
        output:
            - report F1
            - loss 

    '''
    # ========================================
    #               Validation
    # ========================================
    # After the completion of each training epoch, measure our performance on
    # our validation set.

    # Put the model into evaluation mode
    model.eval()
    # Reset the validation loss for this epoch.
    predictions_f1 , true_labels_f1 = [], []
    for batch in tqdm(valid_dataloader, desc = 'Progress Bar'):
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch

        # Telling the model not to compute or store gradients,
        # saving memory and speeding up validation
        with torch.no_grad():
            # Forward pass, calculate logit predictions.
            # This will return the logits rather than the loss because we have not provided labels.
            outputs = model.forward_custom(b_input_ids, b_input_mask, b_labels, token_type_ids=None)
        
        predict_labels = outputs[1]
        label_ids = b_labels.to('cpu').numpy().tolist()
        predictions = []
        for predict_label in predict_labels:
            predictions.append(predict_label)

        for b_input_id, preds, labels in zip(b_input_ids, predictions, label_ids):
            tokens = tokenizer.convert_ids_to_tokens(b_input_id.to('cpu').numpy())

            new_tokens, new_labels, new_preds = [], [], []
            for token, label_idx, pred in zip(tokens, labels, preds):
                if token.startswith("##"):
                    new_tokens[-1] = new_tokens[-1] + token[2:]
                else:
                    new_labels.append(label_idx)
                    new_preds.append(pred)
                    new_tokens.append(token)
            for token, pred, label in zip(new_tokens, new_preds, new_labels):
                predictions_f1.extend([tag_values[pred]])
                true_labels_f1.extend([tag_values[label]])

    if mode == 'train':
        print("Validation F1-Score: {}".format(f1_score(true_labels_f1, predictions_f1,average='macro')))
        print(classification_report(true_labels_f1, predictions_f1,digits = 4))
    elif mode == 'dev':
        labels = tag_values.copy()
        if 'IP' in labels:
            labels.remove('IP')
        if 'SKILL' in labels:
            labels.remove('SKILL')
        if 'PAD' in labels:
            labels.remove('PAD')
        if 'EMAIL' in labels:
            labels.remove('EMAIL')
        print("Validation F1-Score: {}".format(f1_score(true_labels_f1, predictions_f1,labels = labels ,average='macro')))
        print(classification_report(true_labels_f1, predictions_f1,labels = labels ,digits = 4))
    elif mode =='test':
        labels = tag_values.copy()
        if 'IP' in labels:
            labels.remove('IP')
        if 'SKILL' in labels:
            labels.remove('SKILL')
        if 'PAD' in labels:
            labels.remove('PAD')
        print("Validation F1-Score: {}".format(f1_score(true_labels_f1, predictions_f1,labels = labels ,average='macro')))
        print(classification_report(true_labels_f1, predictions_f1,labels = labels ,digits = 4))

################################################################################################################

def valuation_bert_4_sofmax(model, valid_dataloader,tag_values,device,mode):
    '''
        input: 
            - model
            - valid_dataloader
            - tag_values: ['O', 'PER', .... ]
            - device: cuda
        output:
            - report F1
            - loss 

    '''
    # ========================================
    #               Validation
    # ========================================
    # After the completion of each training epoch, measure our performance on
    # our validation set.

    # Put the model into evaluation mode
    model.eval()
    # Reset the validation loss for this epoch.
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    predictions , true_labels = [], []
    for batch in tqdm(valid_dataloader, desc = 'Progress Bar'):
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch

        # Telling the model not to compute or store gradients,
        # saving memory and speeding up validation
        with torch.no_grad():
            # Forward pass, calculate logit predictions.
            # This will return the logits rather than the loss because we have not provided labels.
            outputs = model.forward_custom(input_ids=b_input_ids, attention_mask=b_input_mask, 
                                       labels=b_labels,head_mask=None)
        # Move logits and labels to CPU
        
        logits = outputs[1].detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        # Calculate the accuracy for this batch of test sentences.
        eval_loss += outputs[0].mean().item()
        predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
        true_labels.extend(label_ids)

    eval_loss = eval_loss / len(valid_dataloader)
    print("Validation loss: {}".format(eval_loss))
    pred_tags = [tag_values[p_i] for p, l in zip(predictions, true_labels)
                                 for p_i, l_i in zip(p, l) if tag_values[l_i] != "PAD"]
    valid_tags = [tag_values[l_i] for l in true_labels
                                  for l_i in l if tag_values[l_i] != "PAD"]
    if mode == 'train':
        print("Validation F1-Score: {}".format(f1_score(pred_tags, valid_tags,average='macro')))
        print(classification_report(valid_tags, pred_tags,digits = 4))
    elif mode == 'dev':
        labels = tag_values.copy()
        if 'IP' in labels:
            labels.remove('IP')
        if 'SKILL' in labels:
            labels.remove('SKILL')
        if 'PAD' in labels:
            labels.remove('PAD')
        if 'EMAIL' in labels:
            labels.remove('EMAIL')
        print("Validation F1-Score: {}".format(f1_score(pred_tags, valid_tags,labels = labels ,average='macro')))
        print(classification_report(valid_tags, pred_tags,labels = labels ,digits = 4))
    elif mode =='test':
        labels = tag_values.copy()
        if 'IP' in labels:
            labels.remove('IP')
        if 'SKILL' in labels:
            labels.remove('SKILL')
        if 'PAD' in labels:
            labels.remove('PAD')
        print("Validation F1-Score: {}".format(f1_score(pred_tags, valid_tags,labels = labels ,average='macro')))
        print(classification_report(valid_tags, pred_tags,labels = labels ,digits = 4))

#################################################################################################################
def BERT_EVALUATE(model, tokenizer, dataloader, tag_values, device, type_dataset, model_type):
    if model_type == 'crf':
        valuation_bert_4_crf(model ,tokenizer, dataloader, tag_values , device, type_dataset)
    elif model_type == 'softmax':
        valuation_bert_4_sofmax(model, dataloader, tag_values , device, type_dataset)