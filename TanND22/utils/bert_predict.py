
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
    