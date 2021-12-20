# B_PER = ['anh', 'chị', 'em', 'cô', 'dì', 'gìa', 'bác', 'chú', 'mợ', 'cái', 'thằng']
NAME = ['Mạnh', 'Nghĩa', 'Nam', 'Lợi', 'Tình', 'Hoàng Nguyên', 'Phạm Mạnh']
V = ['đi', 'chơi', 'đang', 'đã', 'ngủ', 'uống', 'học', 'giúp', 'nằm', 'đi', 'đứng', 'yêu', 'ghét', 'kính_trọng']
N = ['cơm', 'du_lịch', 'nhà', 'xe_đạp', 'xe_máy', 'ô_tô', 'taxi', 'con chuột', 'hoa_hồng', 'bàn']

from pyvi import ViTokenizer, ViPosTagger

sequences = []
for name in NAME:
    sent = []
    pos_name = ViPosTagger.postagging(name)[1][0]
    sent.append((name, pos_name, 'B-PER' ))
    for verb in V:
        pos_verb = ViPosTagger.postagging(verb)[1][0]
        sent.append((verb, pos_verb, 'O' ))
        for noun in N:
            pos_noun = ViPosTagger.postagging(noun)[1][0]
            sent.append((noun, pos_noun, 'O' ))
            sequences.append(sent)
            sent.remove((noun, pos_noun, 'O' ))
        sent.remove((verb, pos_verb, 'O' ))
    sent.remove((name, pos_name, 'B-PER' ))




    from numpy import random


import pickle as pk
import pandas as pd

def  augmentation_SiS(infile, outfile, labels):

    '''
    augmentation_SiS: phương pháp augmentation data bằng Shuffle within segments, sử dụng phân phối nhị thức để quyết định shuffle cụm label trong câu
    :infile: path file data gốc (.pkl) 
    :outfile: path file lưu data đã shuffle (.pkl) 
    :labels: các loại nhãn sẽ xet shuffle

    : return: list data đã shuffle
    '''
    print('loading data.....')
    with open(infile, 'rb') as f:
        data =pk.load(f)
    print('complete data')
    data_shuffled = []

    print('shuffling data')
    for index in range(len(data)):
        state = 0
        segment_sentence = []
        d = data[index]
        for jndex in range(len(d)):
            x = d[jndex]
            if x[1] in labels:
                state  = 1
            if jndex == 0:
                segment_sentence.append([x])
            else:
                pre_x = d[jndex - 1]
                if x[1] == pre_x[1]:
                    segment_sentence[-1].append(x)
                else:
                    segment_sentence.append([x])

    
        if state == 1:
            ## shuffle  segment_sentence
            data_shuff = []
            size = len(segment_sentence)
            x = random.binomial(n=1, p=0.5, size=size)

            for index in range(size):

                if x[index] == 1:
                    random.shuffle(segment_sentence[index])

                for sent in segment_sentence[index]:
                    data_shuff.append(sent)
            data_shuffled.append(data_shuff)     
    print('complete shuffle data')

    print('save file shuffled data.......')
    with open(outfile, 'wb') as f:
        pk.dump(data_shuffled, f)
    
    print('done save file!!!!')
    return data_shuffled
            


from numpy import random


def  augmentation_MR(infile, outfile, labels):

    '''
    augmentation_MR: phương pháp augmentation data bằng Shuffle within segments, sử dụng phân phối nhị thức để quyết định shuffle cụm label trong câu
    :infile: path file data gốc (.pkl) 
    :outfile: path file lưu data đã shuffle (.pkl) 
    :labels: các loại nhãn sẽ xet shuffle

    : return: list data đã shuffle
    '''

    new_mentions = []

    print('loading data.....')
    with open(infile, 'rb') as f:
        data =pk.load(f)
    print('complete data')


    vocab = get_vocab(data, labels)

    
    for index in range(len(data)):
        mentions = []

        segment_sentence = segment_data(data[index], labels)
        if segment_sentence is not None:

            size = len(segment_sentence)
            x = random.binomial(n=1, p=0.5, size=size)

            for jndex in range(size):
                ner = segment_sentence[jndex][1]
                if x[jndex] == 1 and ner !='O' :
                    new_word = random.choice(vocab[ner])
                    mentions.append((new_word, ner))
                else:
                    mentions.append(segment_sentence[jndex])
        

            new_mentions.append(mentions)


    
    print('save file data.......')
    with open(outfile, 'wb') as f:
        pk.dump(new_mentions, f)
    
    print('done save file!!!!')

    return new_mentions 


def get_vocab(data, labels):
  
    print('getting vocab....')
    vocab = {}
    for dat in data:
        segment_sent = segment_data(dat, labels)
        if segment_sent is not None:
            for word in segment_sent:
                if word[1] != 'O':
                    if word[1] in vocab:
                        vocab[word[1]].append(word[0])
                    else:
                
                        vocab[word[1]] = [word[0]]

    print('Done etting vocab!!!!!')
    return vocab


def segment_data(data, labels):
    state = 0
    segment_sentence = []
    for jndex in range(len(data)):
        x = data[jndex]
        if labels == 'all':
            state = 1

        elif (x[1] in labels) and (state == 0):
            state = 1

        if jndex == 0:
            segment_sentence.append(x)
        else:
            pre_x = data[jndex - 1]
            if x[1] == pre_x[1]:
                pre_sentence = segment_sentence[-1][0]
                pre_sentence += " " + x[0]
                ner = x[1]

                segment_sentence[-1] = (pre_sentence, ner)
            else:
                segment_sentence.append(x)
    
    if state == 1:
        return segment_sentence
    return None
    