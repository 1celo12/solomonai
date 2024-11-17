import streamlit as st
from solomonai import solomon_ai
st.title("welcome to solomon")
#stores state in a stream lit app
if 'messages' not in st.session_state:
    st.session_state["messages"] = []
# each message is an obj with role and
# conent stored by messages in st.session state
for message in st.session_state["messages"]:
    with st.chat_message(message['role']):
        st.markdown(message['content'])
# create chat block
if prompt := st.chat_input("Hello, how may I help you?"):
    #user role says the message is from a the input from user
    #add last message to message logs using {} obj notation
    st.session_state["messages"].append({"role": "user", "content": prompt})
   
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        ai_response = solomon_ai(st.session_state["messages"])
        st.markdown(ai_response)
        st.session_state["messages"].append({"role": "assistant", "content": ai_response})
