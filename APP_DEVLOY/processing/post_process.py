################################################################################
#hậu xử lý
def quet_dinh_nhan(token):
  all_freq = {}
  word = token[0]
  # print(token[0], token[1])
  if token[0][-1] in [",", ".", ";", "?", "!"] and token[1][-1] == "O" and len(token[1]) > 1:
    ended = (token[0][-1], token[1][-1])
    for i in token[1][:-1]:
      if i in all_freq:
          all_freq[i] += 1
      else:
          all_freq[i] = 1
    res = max(all_freq, key = all_freq.get)
    return [(token[0][:-1], res), ended]
  else:
    for i in token[1]:
        if i in all_freq:
            all_freq[i] += 1
        else:
            all_freq[i] = 1
    res = max(all_freq, key = all_freq.get)
    return [(token[0], res)]


import re

def is_URL(token):
    index = 0
    indexs = []
    for word in token.split(" "):
        # print(word)
        domain = re.findall(r'\b((?:https?://)?(?:(?:www\.)?(?:[\da-z\.-]+)\.(?:[a-z]{2,6})|(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)|(?:(?:[0-9a-fA-F]{1,4}:){7,7}[0-9a-fA-F]{1,4}|(?:[0-9a-fA-F]{1,4}:){1,7}:|(?:[0-9a-fA-F]{1,4}:){1,6}:[0-9a-fA-F]{1,4}|(?:[0-9a-fA-F]{1,4}:){1,5}(?::[0-9a-fA-F]{1,4}){1,2}|(?:[0-9a-fA-F]{1,4}:){1,4}(?::[0-9a-fA-F]{1,4}){1,3}|(?:[0-9a-fA-F]{1,4}:){1,3}(?::[0-9a-fA-F]{1,4}){1,4}|(?:[0-9a-fA-F]{1,4}:){1,2}(?::[0-9a-fA-F]{1,4}){1,5}|[0-9a-fA-F]{1,4}:(?:(?::[0-9a-fA-F]{1,4}){1,6})|:(?:(?::[0-9a-fA-F]{1,4}){1,7}|:)|fe80:(?::[0-9a-fA-F]{0,4}){0,4}%[0-9a-zA-Z]{1,}|::(?:ffff(?::0{1,4}){0,1}:){0,1}(?:(?:25[0-5]|(?:2[0-4]|1{0,1}[0-9]){0,1}[0-9])\.){3,3}(?:25[0-5]|(?:2[0-4]|1{0,1}[0-9]){0,1}[0-9])|(?:[0-9a-fA-F]{1,4}:){1,4}:(?:(?:25[0-5]|(?:2[0-4]|1{0,1}[0-9]){0,1}[0-9])\.){3,3}(?:25[0-5]|(?:2[0-4]|1{0,1}[0-9]){0,1}[0-9])))(?::[0-9]{1,4}|[1-5][0-9]{4}|6[0-4][0-9]{3}|65[0-4][0-9]{2}|655[0-2][0-9]|6553[0-5])?(?:/[\w\.-]*)*/?)\b', word)
        
        if len(domain) != 0:
            index_start_domain = word.find(domain[0]) + index
            if word.find(domain[0]) == 0:
                index_end_domain = index_start_domain + len(word)
            else:
                index_end_domain = index_start_domain + len(domain[0])
            indexs.append((index_start_domain, index_end_domain))
        index += len(word) + 1
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
    



from pyvi import ViTokenizer, ViPosTagger

def hau_xu_ly(sent, out):
  token, label = list(zip(*out))
  parts = sent.split()
  datas = []
  count = 0

  for i in range(len(parts)):
    word = parts[i]
    # print(word, count)
    try:
      
      # print(word, count)
      word_updated, index = gheptu(word,token , count)
    except:
    #  print(word_updated)
     print('error: {}, {}, {}'.format(word,token , count))
     break

    out_qdn = quet_dinh_nhan((word_updated, label[count: index]))
    
    for o in out_qdn:
      w, label_merged = o
      datas.append((w, label_merged))
    count = index


  datas_trained = preprocess_email_url(datas)
  # print(datas_trained)
  gomcum = gom_cum(datas_trained)
  # print(gomcum)
  if len(gomcum) != 0:
    indexs = cluster(gomcum, 2)
    # print(indexs)
    for index in indexs:
      string, label = list(zip(*datas_trained[index[0]: index[-1] + 1]))
      # string_loc = " ".join(string)
      if is_ADDRESS(string, label) == True:
        for i in range(index[0], index[-1] + 1):
          # print('hdhdhd')
          datas_trained[i] =(datas_trained[i][0], "ADDRESS")
  return datas_trained
# else:


def gom_cum(tokens):
  indexs = []
  for index in range(len(tokens)):
    token = tokens[index]
    if token[1] == "LOCATION" or token[1] == "ADDRESS" :
      indexs.append(index)
  return indexs

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
  

def gheptu(word, parts, index):
  
  if parts[index] == "[UNK]":
    return word, index + 1
    
  start = index
  # print(word)
  for i in range(start, len(parts) + 1):
    end = i
    # print("".join(parts[start:end]))
    # print(word.lower(),"".join(parts[start:end]).lower())



    if word.lower() == "".join(parts[start:end]).lower():
      # print('oke')
      # print(word, end)
      return word, end
  # print('fuck')
  # print(word, index)
  return None

def has_numbers(inputString):
  parts = inputString.split()
  # print(parts)
  for i in range(len(parts)):
    part = parts[i]
    for char in part:
      if char.isdigit():
        # print(i)
        if i > 0 and parts[i-1].lower() in ["quận", "q."]:
          return False
        else:
          return True
  return False

def is_ADDRESS(string, label):
  index_dau = [i for i, e in enumerate(string) if e == ","]
  index_not_dau_phay = [i for i, e in enumerate(label) if e == "O"]

  uy_tin = 0
  string_loc = " ".join(string)
  # print(label)
 
  # print(string)
  if 'ADDRESS' in label:
    uy_tin += 0.2
  
  # if '(' in string_loc or ')' in string_loc:
  #   uy_tin -= 0.025
  

  if has_numbers(string_loc):
    uy_tin += 0.05
  
  count =  len(index_dau) 

  # count = label.count('LOCATION')
  # print(count)
  if count > 0 and count < 3:  #count = 1, 2:
    uy_tin += 0.15
  elif count > 2:
    uy_tin += 0.2
  
  for i in index_not_dau_phay:
      if string[i] != ",":
        uy_tin -= 0.05
  level = ["số", "đường","tổ", "ngõ", "toà", "ngách", "hẻm","kiệt", "chung_cư", "ấp" ,"thôn", "khu","phố" , "quận", "phường", "xã", "thị_xã","huyện", "thành", "tp", "tỉnh" ]
  level_0 ={'status': True,'keywords': ["toà", "chung_cư", "số"] }
  level_1 = {'status': True, 'keywords': ["đường", "ngõ", "ngách", "hẻm","kiệt",]}
  level_2 = {'status': True, 'keywords':["ấp" ,"thôn", "khu","phố" , "quận", "phường", "xã", "tổ", "dân_phố"]}
  level_3 = {'status': True,'keywords':["thị","huyện"]}
  level_4 = {'status': True,'keywords':["thành", "tp", "tỉnh"]}

  parts =  ViPosTagger.postagging(ViTokenizer.tokenize(string_loc))[0]

  for seg_word in parts:
    # print(seg_word)
    if seg_word.lower() in level:
 

      if seg_word.lower() in level_0['keywords'] and level_0['status'] == True:
        uy_tin += 0.125
        level_0['status'] = False

      if seg_word.lower() in level_1['keywords'] and level_1['status'] == True:
        uy_tin += 0.125
        level_1['status'] = False

      elif seg_word.lower()  in level_2['keywords'] and level_2['status'] == True:
        uy_tin += 0.1
        level_2['status'] = False
      elif seg_word.lower() in  level_3['keywords'] and level_3['status'] == True:
   
        uy_tin += 0.05
        level_3['status'] = False
      elif seg_word.lower() in level_4['keywords'] and level_4['status'] == True:
     
        uy_tin += 0.025
        level_4['status'] = False

      
      # print(word.lower(), level_1.index(word.lower()) + 1)
      

  # print("check{}".format(uy_tin))
  if uy_tin >= 0.3:
    return True
  else:
    return False