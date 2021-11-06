Named Entity Recognition using BiLSTM-CRF

--data: vlsp_2018, format [[word, tag],[word, tag],...,[word, tag]]

--crf.py: custom layer CRF

--model.py: model architecture module

--utils: preprocessing data, visualize module

--val: evaluate test data module

--bilstm-crf.ipynb: training model and save best weight

--testing: evaluate test data and real data, and save analysis result by .csv format, just only error sentence.

--vocabs: file vocabs for this data

--analysis_test.csv, analysys_train.csv: file for analysis error.
