import streamlit as st
from src.wrapper import Wrapper

w = Wrapper('checkpoint/20200813-235524/model.h5','data/token2id.json')

# streamlit 
st.header("Movie Comment Classification Demo")

text = st.text_input("Enter comment")
if st.button("Run", key="text"):
    res = w.predict(text.title())

    print(res)
    if res[0] > res[1]:
        st.success('negative')
    else:
        st.success('positive')
   
