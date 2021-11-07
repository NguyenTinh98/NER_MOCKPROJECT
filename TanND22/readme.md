# BERT
## model_save
    - /2018
        - /BIO_BERT_MULTI.pt
        - /IO_BERT_MULTI.pt
## file_pkl
    - Lưu các file phục vụ cho việc check lỗi (không còn dùng nhiều)
## data_check_err
    - Chứa các file để check lỗi 
## vlsp2018
    - dataset của vlsp2018
## utils
    - bert_evaluate.py
        + valuation_bert_multi : dùng để đánh giá mô hình BERT_MULTI thôi nha
    - bert_load.py: chứa các hàm cần thiết để load data và preprocessing data cho bert (nói chung luôn)
    - bert_predict.py: 
        +  predict_text: dùng để dự đoán và giúp in ra file check lỗi cho mô hình
    - check_err.py: Chứa các file của sửa lỗi của Mạnh, phục vụ cho mục đích kiểm tra lỗi của mô hình
    - visual_predict.py
        + BERT_PREDICT: dùng để dự đoán mô hình với dữ liệu thực tế, có thể dùng **CPU** hoặc **GPU**
