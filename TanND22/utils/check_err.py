import pandas as pd
import numpy as np

def check_sentence(num_sent, y_pred, x_test):
    """
    
    function get_sentence_pred: kiểm tra 1 câu những label được dự đoán đúng (is_error = 0), dự đoán sai (is_error =1)
    :num_sent: thể hiện số thứ tự của câu trong 1 list X_test
    :y_pred: input đã được dự đoán từ model với input là x_test
    :x_test: câu được cho vào để dự đoán dạng list tuple, mỗi tuple là: (word, pos, ner)
    :output: trả về tuple (num_sent, word, pos, ner, ner_pred, is_error)
    """

    sentence_info = []
    sent_error = False
    for index in range(len(x_test)):
        word_info = x_test[index]
        
        word = word_info[0]
        pos = word_info[1]
        true_ner = word_info[2]
     
        ner_pred = y_pred[index]

        if true_ner == y_pred[index]:
            is_error = 0
            sentence_info.append((num_sent, index, word, pos, true_ner, ner_pred, is_error))
       
        else:
            is_error = 1
            sent_error = True
            sentence_info.append((num_sent, index, word, pos, true_ner, ner_pred, is_error))

    return  sentence_info, sent_error


def convert_sentences_to_df(Y_pred, X_test, option = "full"):

    """
    function get_sentences_pred: kiểm tra N câu những label được dự đoán đúng (is_error = 0), dự đoán sai (is_error =1)
    : option: "full", "ony_true", "only_false"
    :Y_pred: input đã được dự đoán từ model với input là X_test
    :X_test: N câu được cho vào để dự đoán
    :output: trả về dataframe có các columns (num_sent, word, pos, ner, ner_pred, is_error)
    """

    df = pd.DataFrame([], columns = ["#sent","#word","word", "true_pos", "true_ner", "predict_ner", "is_error"])
    
    for index in range(len(X_test)):

        x_test = X_test[index]
        y_pred = Y_pred[index]
        sentence_info, sent_error = check_sentence(index, y_pred, x_test)
        if option == "full":
            df = df.append(pd.DataFrame(sentence_info,  columns = ["#sent","#word", "word", "true_pos", "true_ner", "predict_ner", "is_error"]))
        
        if option == "ony_true":
            if sent_error == False:
                df = df.append(pd.DataFrame(sentence_info,  columns = ["#sent", "#word", "word", "true_pos", "true_ner", "predict_ner", "is_error"]))

        if option == "ony_false":
            if sent_error == True:
                df = df.append(pd.DataFrame(sentence_info,  columns = ["#sent", "#word", "word", "true_pos", "true_ner", "predict_ner", "is_error"]))


        
    return df.copy()
    
    

def  analyst_error(num_sent, y_pred, x_test):
    """
    function make_error: phân tích lỗi các từ xung quanh từ bị lỗi
    :Y_pred: input đã được dự đoán từ model với input là x_test
    :X_test:  câu được cho vào để dự đoán
    :output: trả về tuple có các columns (num_sent, words, poss, ners, ner_preds, is_error)
    """
    
    sentence_info = []
    for index in range(len(x_test)):
     
#         print(x_test[index][2] , y_pred[index])
        if x_test[index][1] != y_pred[index]:
             
            
            if index == 0:
                sentence_info.append((num_sent, 
                  [x_test[index][0], x_test[index + 1][0], x_test[index + 2][0]],
                  [x_test[index][2], x_test[index + 1][2], x_test[index + 2][2]], 
                  [x_test[index][1], x_test[index + 1][1], x_test[index + 2][1]], 
                  [y_pred[index],y_pred[index + 1],y_pred[index + 2]],
                     x_test[index][1]))
            
            if index >= 1 and index !=len(x_test) -1  :
                sentence_info.append((num_sent, 
                  [x_test[index - 1][0], x_test[index][0], x_test[index + 1][0]],
                  [x_test[index - 1][2], x_test[index][2], x_test[index + 1][2]], 
                  [x_test[index - 1][1], x_test[index][1], x_test[index + 1][1]], 
                  [y_pred[index - 1],y_pred[index],y_pred[index + 1]],
                                      x_test[index][1]))
          
            if index ==len(x_test) - 1  :
                    sentence_info.append((num_sent, 
                      [x_test[index - 2][0], x_test[index][0], x_test[index][0]],
                      [x_test[index - 2][2], x_test[index][2], x_test[index][2]], 
                      [x_test[index - 2][1], x_test[index][1], x_test[index ][1]], 
                      [y_pred[index - 2],y_pred[index-1],y_pred[index]],
                                          x_test[index][1]))
    return   sentence_info 
    

def analyts_errors(Y_pred, X_test):
    """
    Phân tích lỗi theo cụm từ gần nhất, gồm 2 từ xung quanh 
    :Y_pred: outputs của X_test
    : X_test: m câu dự đoán
    :output: return dataframe chứa thông tin các từ xung quanh từ bị "dự đoán sai", hiện tại: 2 từ xung quanh
    """
    df = pd.DataFrame([], columns = ["#sent","words", "true_poss", "true_ners", "predict_ners", "error_labels"])
    sentences_info = []
    for index in range(len(X_test)):
        x_test = X_test[index]
        y_pred = Y_pred[index]
        sentence_info = analyst_error(index, y_pred, x_test)
        df = df.append(pd.DataFrame(sentence_info,  columns = ["#sent","words", "true_poss", "true_ners", "predict_ners","error_labels"]))
    return df.copy()   