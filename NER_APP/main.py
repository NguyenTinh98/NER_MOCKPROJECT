import streamlit as st 
from utils import *
import torch
from transformers import AutoTokenizer
from transformers import  BertModel, BertConfig
from model import *
import time

@st.cache(allow_output_mutation=True)
def load_model(PATH = 'model/model_best.pt'):
    print("1. Dowload some package")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False,use_fast=False)
    config = BertConfig.from_pretrained('bert-base-multilingual-cased', output_hidden_states=True)
    config.max_position_embeddings = 512
    bert_model = BertModel.from_pretrained(
                        'bert-base-multilingual-cased',
                        config=config,
                        add_pooling_layer=False
        )
    print("1. Dowload Done")

    print('2.Load model')

    tag_values = ['EMAIL', 'ADDRESS','PERSON','PHONENUMBER','MISCELLANEOUS','QUANTITY','PERSONTYPE',
                'ORGANIZATION','PRODUCT','SKILL','IP','LOCATION','O','DATETIME','EVENT','URL']

    model = BERT_SOFTMAX(bert_model, num_labels=len(tag_values)+1)
    model.load_state_dict(torch.load(PATH), strict=False)
    model.to(device)
    print('2.Load model Done')
    return tokenizer, model, tag_values, device

tokenizer, model, tag_values ,device = load_model()

HTML_WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem">{}</div>"""
def main():
    """NER Streamlit App"""

    st.title("Name Entity Checker")

    activities = ["NER Checker","NER For PDF"]
    choice = st.sidebar.selectbox("Select Activity",activities)

	
    if choice == 'NER Checker':
        
        st.subheader("Named Entity Recog with Spacy")
        raw_text = st.text_area("Enter Text Here","Type Here")
        if st.button("Analyze") and len(raw_text)>0:
            start = time.time()
            doc = predict_text(model, tokenizer, tag_values ,raw_text,device)
            html = visualize_spacy(doc)
            html.replace("\n\n","\n")
            st.title('Streamlit example')
            HTML_WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem">{}</div>"""
            st.write(HTML_WRAPPER.format(html),unsafe_allow_html=True)
            t = f'{time.time() - start:.5}'
            st.write("Time processing: "+ t + " s")
    if choice == 'NER For PDF':
        st.subheader("Named Entity Recog with PDF")
if __name__ == '__main__':
	main()