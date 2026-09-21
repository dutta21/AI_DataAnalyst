import streamlit as st
from groq import Groq
from dotenv import load_dotenv
import os

from router import route_query


load_dotenv()

# Fetch the data
secret_key = os.getenv("API_KEY")

st.set_page_config(
    page_title="AI Chat",
    page_icon="🤖"
)

st.title("🤖 AI Assistant")

client = Groq(
    api_key=secret_key
)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "system",
            "content": "You are a helpful AI assistant."
        }
    ]

# Display previous messages
for message in st.session_state.messages:
    if message["role"] != "system":
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# User input
if prompt := st.chat_input("Ask anything..."):

    # Show user message
    with st.chat_message("user"):
        st.markdown(prompt)

    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    # Route through the router: it decides direct answer vs. news lookup
    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):
            answer = route_query(client, st.session_state.messages)

        st.markdown(answer)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })