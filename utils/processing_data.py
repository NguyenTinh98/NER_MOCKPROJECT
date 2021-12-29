import string
import unicodedata
from pyvi import ViTokenizer, ViPosTagger
import re


def preprocess_email_url(datas):
  datas_trained = []
  for i in range(len(datas)):
    data = datas[i]

    if data[1] == 'EMAIL':
      check = is_Email(data[0])
      if len(check) == 0:
        data = (data[0], 'O')
    
    if data[1] != 'EMAIL' and  data[1] != 'URL': #(url, org, loc, o,.....)
      check = is_Email(data[0])
      if len(check) > 0:
        data = (data[0], 'EMAIL')

  

    if data[1] == "URL":
      # print(data[0])
      check = is_URL(data[0])
      if len(check) > 0 and  check[0][1] - check[0][0] == len(data[0]):
        data = (data[0], 'URL')
      else: 
        data = (data[0], 'O')
      
    try:
      if data[1] != 'URL' and data[1] != 'EMAIL':
        check = is_URL(data[0])
        if len(check) > 0 and  check[0][1] - check[0][0] == len(data[0]):
          data = (data[0], 'URL')
    except:
      print(check)
    datas_trained.append(data)
  return datas_trained
    
# sent = 'pham van manh have email ( pvm26042000@gmail.com ) ....'
# out = [('pham', 'O'), ('van', 'O'), ('manh', 'O'), ('have', 'O'), ('email', 'O'), ('(', 'O'),  ('pvm26042000', 'EMAIL'), ('@', 'EMAIL'),('gmail', 'EMAIL'), ('.', 'EMAIL'),('com', 'EMAIL'),(')', 'O'),('....', 'O')]

def merge_word(sent, pred_out):
  '''
    :sent: is input sentences (hanlded pre-processing). example: 'pham van manh have email ( pvm26042000@gmail.com ) ....'
    :out : is input of predict, is list tuple. example: [('pham', 'O'), ('van', 'O'), ('manh', 'O'), ('have', 'O'), ('email', 'O'), ('(', 'O'),  ('pvm26042000', 'EMAIL'), ('@', 'EMAIL'),('gmail', 'EMAIL'), ('.', 'EMAIL'),('com', 'EMAIL'),(')', 'O'),('....', 'O')]
  '''
  out_merged = []
  parts = sent.split()
  for index in range(0, len(parts)):
    word = parts[index]

    
    for jndex in range(1, len(pred_out) + 1):
      token = pred_out[0:jndex]
      ws_token, _ = list(zip(*token))
      word_token = "".join(ws_token)
   
      if word_token == word:
        if len(token) == 1:
          out_merged.append(token[0])
        elif len(token) > 1:
          a, b = list(zip(*token))
          word_merged = "".join(a)
          l_merged = decide_label((word_merged, b))
          out_merged.append(l_merged)
        pred_out = pred_out[jndex:]
        break
  return out_merged

def post_processing(origin_sentence, out_predict):

  out_merged = merge_word(origin_sentence, out_predict)
  datas_trained = post_process_email_url(out_merged)

  indexs = []
  for index in range(len(datas_trained)):
    token = datas_trained[index]
    if token[1] == "LOCATION" or token[1] == "ADDRESS" :
      indexs.append(index)

  if len(indexs) != 0:
    gr_indexs = cluster(indexs, 3)


    for index in gr_indexs:
      string, label = list(zip(*datas_trained[index[0]: index[-1] + 1]))
      # print(string, label)
      if is_ADDRESS(string, label) == True:
        for i in range(index[0], index[-1] + 1):
          datas_trained[i] =(datas_trained[i][0], "ADDRESS")
      else:
        for i in range(index[0], index[-1] + 1):
          if datas_trained[i][0] == ',':
            datas_trained[i] = (datas_trained[i][0], "O")
          else:
            datas_trained[i] =(datas_trained[i][0], "LOCATION")
  return datas_trained

def cluster(data, maxgap):
    '''Arrange data into groups where successive elements
       differ by no more than *maxgap*
        >>> cluster([1, 6, 9, 100, 102, 105, 109, 134, 139], maxgap=10)
        [[1, 6, 9], [100, 102, 105, 109], [134, 139]]
        >>> cluster([1, 6, 9, 99, 100, 102, 105, 134, 139, 141], maxgap=10)
        [[1, 6, 9], [99, 100, 102, 105], [134, 139, 141]]
    '''
    data.sort()
    groups = [[data[0]]]
    for x in data[1:]:
        if abs(x - groups[-1][-1]) <= maxgap:
            groups[-1].append(x)
        else:
            groups.append([x])
    return groups
  


def has_numbers(inputString):
  parts = inputString.split()
  for i in range(len(parts)):
    part = parts[i]
    for char in part:
      if char.isdigit():
        if i > 0 and parts[i-1].lower() in ["quận", "q."]:
          return False
        else:
          return True
  return False

def is_ADDRESS(string, label):

  uy_tin = 0
  string_loc = " ".join(string)

  level = ["số", "lô", "km","quốc_lộ","đại_lộ","kcn", "đường","tổ", "ngõ", "toà", "ngách", "hẻm","kiệt", "chung_cư", "số_nhà","ấp" ,"thôn", "khu","phố" , "quận", "phường", "xã", "thị_xã","huyện", "thành_phố", "tp", "tỉnh" ]
  level_0 ={'status': True,'keywords': ["toà", "chung_cư", "số", "lô", "số_nhà"] }
  level_1 = {'status': True, 'keywords': [ "ngõ", "ngách", "hẻm","kiệt","kcn", "km"]}
  level_2 = {'status': True, 'keywords':["ấp" ,"thôn", "khu","phố" , "quận", "phường", "xã", "tổ", "dân_phố", "đường", "quốc_lộ", "đại_lộ"]}
  level_3 = {'status': True,'keywords':["thị","huyện"]}
  level_4 = {'status': True,'keywords':["thành_phố", "tp", "tỉnh"]}

  parts =  ViPosTagger.postagging(ViTokenizer.tokenize(string_loc))[0]

  for index in range(len(parts)):
    seg_word = parts[index]
    if index == 0 and  has_numbers(seg_word.split(" ")[0]):
        uy_tin += 0.3
        break

    if seg_word.lower() in level:
 
      if seg_word.lower() in level_0['keywords'] and level_0['status'] == True:
        uy_tin += 0.3
        break

      elif seg_word.lower() in level_1['keywords'] and level_1['status'] == True:
        uy_tin += 0.25
        level_1['status'] = False

      elif seg_word.lower()  in level_2['keywords'] and level_2['status'] == True:
        uy_tin += 0.025
        level_2['status'] = False
      elif seg_word.lower() in  level_3['keywords'] and level_3['status'] == True:
   
        uy_tin += 0.015
        level_3['status'] = False
      elif seg_word.lower() in level_4['keywords'] and level_4['status'] == True:
     
        uy_tin += 0.01
        level_4['status'] = False
  print(uy_tin)
  if uy_tin >= 0.29:
    return True
  
  return False


def decide_label(part):
  word = part[0]
  labels = part[1]
  return (word, max(labels))


import re
def constain_alpha(token):

  for character in token:

    is_letter = character.isalpha()
    if is_letter == True:
      return True
  
  return False

def is_URL(token):
    token = token.lower()
    index = 0
    indexs = []
    if constain_alpha(token) == True:
 
      
      # print(word)
      domain = re.findall(r'\b((?:https?://)?(?:(?:www\.)?(?:[\da-z\.-]+)\.(?:[a-z]{2,6})|(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)|(?:(?:[0-9a-fA-F]{1,4}:){7,7}[0-9a-fA-F]{1,4}|(?:[0-9a-fA-F]{1,4}:){1,7}:|(?:[0-9a-fA-F]{1,4}:){1,6}:[0-9a-fA-F]{1,4}|(?:[0-9a-fA-F]{1,4}:){1,5}(?::[0-9a-fA-F]{1,4}){1,2}|(?:[0-9a-fA-F]{1,4}:){1,4}(?::[0-9a-fA-F]{1,4}){1,3}|(?:[0-9a-fA-F]{1,4}:){1,3}(?::[0-9a-fA-F]{1,4}){1,4}|(?:[0-9a-fA-F]{1,4}:){1,2}(?::[0-9a-fA-F]{1,4}){1,5}|[0-9a-fA-F]{1,4}:(?:(?::[0-9a-fA-F]{1,4}){1,6})|:(?:(?::[0-9a-fA-F]{1,4}){1,7}|:)|fe80:(?::[0-9a-fA-F]{0,4}){0,4}%[0-9a-zA-Z]{1,}|::(?:ffff(?::0{1,4}){0,1}:){0,1}(?:(?:25[0-5]|(?:2[0-4]|1{0,1}[0-9]){0,1}[0-9])\.){3,3}(?:25[0-5]|(?:2[0-4]|1{0,1}[0-9]){0,1}[0-9])|(?:[0-9a-fA-F]{1,4}:){1,4}:(?:(?:25[0-5]|(?:2[0-4]|1{0,1}[0-9]){0,1}[0-9])\.){3,3}(?:25[0-5]|(?:2[0-4]|1{0,1}[0-9]){0,1}[0-9])))(?::[0-9]{1,4}|[1-5][0-9]{4}|6[0-4][0-9]{3}|65[0-4][0-9]{2}|655[0-2][0-9]|6553[0-5])?(?:/[\w\.-]*)*/?)\b', token)
      
      if len(domain) != 0:
          index_start_domain = token.find(domain[0]) + index
          if token.find(domain[0]) == 0:
              index_end_domain = index_start_domain + len(token)
          else:
              index_end_domain = index_start_domain + len(domain[0])
          indexs.append((index_start_domain, index_end_domain))
      index += len(token) + 1
    return indexs

def is_Email(token):
    index = 0
    indexs = []
    for word in token.split(" "):
        # print(word)
        emails = re.findall(r"[\w.+-]+@[\w-]+\.[\w.-]+", word)
        # print(emails)
        if len(emails) != 0:
            index_start_email = word.find(emails[0]) + index
            
            index_end_email = index_start_email + len(emails[0])
            
            indexs.append((index_start_email, index_end_email))
        index += len(word) + 1
    return indexs
def is_IP(token):
  index = 0
  indexs = []
  for word in token.split(" "):
      # print(word)
      emails = re.findall(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b", word)
      # print(emails)
      if len(emails) != 0:
          index_start_email = word.find(emails[0]) + index
          
          index_end_email = index_start_email + len(emails[0])
          
          indexs.append((index_start_email, index_end_email))
      index += len(word) + 1
  return indexs

def post_process_email_url(datas):
  black_word = ["tp.hcm"]
  datas_trained = []
  for i in range(len(datas)):
    data = datas[i]

      # check predict email
    if data[1] == 'EMAIL':
        check = is_Email(data[0])
        if len(check) == 0:
          data = (data[0], 'O')
    
    elif data[1] == 'URL':
        check = is_URL(data[0])

        if len(check) == 0 or  check[0][1] - check[0][0]!= len(data[0]):
        
          data = (data[0], 'O')
    
    elif data[1] == 'IP':
        check = is_IP(data[0])
        if len(check) == 0 or  check[0][1] - check[0][0]!= len(data[0]):
          if data[0].isalnum():
            data = (data[0], 'QUANTITY')
          else:
            data = (data[0], 'O')

          # return
    if data[1] in ['O'] and data[1].lower() not in black_word:
        # print(data[0])
        check_url = is_URL(data[0])
        check_email= is_Email(data[0])

        if len(check_url) > 0 and  check_url[0][1] - check_url[0][0] == len(data[0]):

          data = (data[0], 'URL')

        elif len(check_email) > 0:
          data = (data[0], 'EMAIL')
      
    datas_trained.append(data)
  return datas_trained