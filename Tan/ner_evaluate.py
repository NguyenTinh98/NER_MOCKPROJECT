import torch
from tqdm import tqdm
from pyvi import ViTokenizer
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
import numpy as np
from keras.preprocessing.sequence import pad_sequences

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
#     print("Validation Accuracy: {}".format(accuracy_score(pred_tags, valid_tags)))
    print("Validation F1-Score: {}".format(f1_score(pred_tags, valid_tags,average='micro')))
    print(classification_report(valid_tags, pred_tags,digits = 4))
    print("####################")




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

def predict_text(model, tokenizer, tag_values,test_sentence):
    #predict with model
    model.eval()
    test_sentence_token = transform_test(test_sentence, 'word')
    subwords = tokenize_predict(tokenizer, test_sentence_token)
    input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(subwords)],
                              maxlen=512, dtype="long", value=0.0,
                              truncating="post", padding="post")
    input_ids_tensor = torch.tensor(input_ids).type(torch.LongTensor).cuda() #Fixfug here
    input_mask = [[float(i != 0.0) for i in ii] for ii in input_ids]
    input_mask_tensor = torch.tensor(input_mask).type(torch.LongTensor).cuda() 
    with torch.no_grad():
        outputs = model(input_ids_tensor, token_type_ids = None, attention_mask = input_mask_tensor)
        
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
    