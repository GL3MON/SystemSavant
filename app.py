import streamlit as st
from SystemSavant.pipeline.inference import Chat
Chat_Agent = Chat()
Chat_Agent.initiate()

st.title('System Savant')
st.header('Your very own personal Computer Organization and Architecture Tutor')
query = st.text_input("Enter Your Question Below")
if (st.button("Submit")):
    st.info("System Savant is thinking...")
    st.code(Chat_Agent.respond(query))
    st.info("System Savant has responded to your query")