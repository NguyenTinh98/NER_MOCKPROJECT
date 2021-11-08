import pandas as pd
import numpy as np


def check_sentence(y_pred, x_test, n_col = 3, num_sent = 0):
    """
    
    function check_sentence: kiểm tra 1 câu với những label (word) được dự đoán đúng (is_error = 0), dự đoán sai (is_error =1)
    :num_sent: thể hiện thứ tự (index) của câu trong 1 list X_test, default = 0
    :y_pred: input đã được dự đoán từ model với input là x_test
    :x_test: câu được cho vào để dự đoán dạng list tuple, mỗi tuple là: (word, pos, ner)
    :output: trả về tuple (num_sent, word, pos, ner, ner_pred, is_error), sent_error (return sent có đúng hoàn toàn hay sai 1 phần)
    """

    sentence_info = []
    sent_error = False
    for index in range(len(x_test)):
        word_info = x_test[index]
        
        if n_col == 3:
            word = word_info[0]
            pos = word_info[1]
            true_ner = word_info[2]
        if n_col == 2:
            word = word_info[0]
            pos = "?"
            true_ner = word_info[1]
     
        ner_pred = y_pred[index]

        if true_ner == y_pred[index]:
            is_error = 0
            sentence_info.append((num_sent, index, word, pos, true_ner, ner_pred, is_error))
       
        else:
            is_error = 1
            sent_error = True
            sentence_info.append((num_sent, index, word, pos, true_ner, ner_pred, is_error))

    return  sentence_info, sent_error


def convert_sentences_to_df(Y_pred, Data_test, option = "full", n_col = 3):

    """
    function convert_sentences_to_df: kiểm tra N câu những label được dự đoán đúng (is_error = 0), dự đoán sai (is_error =1)
    : option: "full": tất cả , "ony_true": chỉ lấy câu đúng, "only_false": chỉ lấy câu sai
    :Y_pred: input đã được dự đoán từ model với input là X_test
    :Data_test: N câu được cho vào để dự đoán, một list các sentences dạng [[(word, pos, ner), (word1, pos1, ner1)], [(word, pos, ner), (word1, pos1, ner1)]]
    :output: trả về dataframe có các columns (num_sent,numword, word, pos, ner, ner_pred, is_error)
    """
    
    

    df = pd.DataFrame([], columns = ["#sent","#word","word", "true_pos", "true_ner", "predict_ner", "is_error"])
    
    for index in range(len(Data_test)):

        x_test = Data_test[index]
        y_pred = Y_pred[index]
        sentence_info, sent_error = check_sentence(y_pred, x_test, n_col = n_col, num_sent = index)
        if option == "full":
            df = df.append(pd.DataFrame(sentence_info,  columns = ["#sent","#word", "word", "true_pos", "true_ner", "predict_ner", "is_error"]))
        
        if option == "only_true":
            if sent_error == False:
                df = df.append(pd.DataFrame(sentence_info,  columns = ["#sent", "#word", "word", "true_pos", "true_ner", "predict_ner", "is_error"]))

        if option == "only_false":
            if sent_error == True:
                df = df.append(pd.DataFrame(sentence_info,  columns = ["#sent", "#word", "word", "true_pos", "true_ner", "predict_ner", "is_error"]))


        
    return df.copy()

# find word index near
def get_data_error(df, num_sent, dist_size):
    """
    function get_data_error: phân tích lỗi xung quanh từ bị lỗi
    df: DataFrame để phân tích gồm các columns (#sent, #word, word, true_pos, true_ner,predict_ner,is_error)
    dist_size: khoảng cách xung quanh từ bị lỗi
    num_sent: câu nằm trong câu
    """
    is_error = 1
    col_name = "#sent"
    data_test = df
    df_condition = data_test.loc[(data_test[col_name] ==  num_sent)]
    df_condition_is_error = df_condition.loc[df_condition["is_error"] == is_error]
    
    if len(df_condition_is_error) ==  0:
        return  print("Not exited data row by condition this")

    else:
        
        word_near = df_condition_is_error["#word"].iloc[0]
        word_least = df_condition_is_error["#word"].iloc[-1]
        max_len = df_condition["#word"].iloc[-1]

        if word_near <= dist_size:
            word_index_left = 0
        else:
            word_index_left = word_near - dist_size 

        if dist_size >= max_len - word_least:
            word_index_right = -1

        else:
            word_index_right = word_least + dist_size


        if word_index_right == -1:
            rs_df = df_condition.iloc[word_index_left:]
        else:
            rs_df = df_condition.iloc[word_index_left:word_index_right + 1]
        return rs_df