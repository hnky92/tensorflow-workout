import streamlit as st
from src.wrapper import Wrapper

# initializer model
config = { 
        "src_model":"checkpoints/1584974736/model-12400",
        "src_char2id":"data/char2id.txt",
        "max_len":150
        }

w = Wrapper(config["src_model"], config["src_char2id"], config["max_len"])

# streamlit 
st.header("Movie Comment Classification Demo")

text = st.text_input("Enter comment")
if st.button("Run", key="text"):
    res = w.run_batch([text.title()])
    if res[0] == 0:
        st.success('negative')
    else:
        st.success('positive')
