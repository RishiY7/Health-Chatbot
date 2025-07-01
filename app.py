# app.py

# HealthBot - Simple Health Chat App using Transformers and Streamlit
# Created by Rishi Yadav for educational/demo purposes

import streamlit as st
from transformers import pipeline

# Load the context file (this contains all the health info)
with open("context.txt", "r", encoding="utf-8") as f:
    context = f.read()

# Load a pretrained QA model from Hugging Face
qa_pipeline = pipeline("question-answering")

# Streamlit page setup
st.set_page_config(page_title="🩺 Health-Chatbot", layout="wide")
st.title("🩺 Health Chatbot")
st.markdown("Ask any health-related question and get instant guidance!")

# Store conversation history
if "chat" not in st.session_state:
    st.session_state.chat = []

# Display chat history
for msg in st.session_state.chat:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Take user input
user_question = st.chat_input("Enter your health question...")

if user_question:
    # Show user's question
    st.session_state.chat.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.write(user_question)

    # Generate model answer from context
    with st.spinner("Thinking..."):
        try:
            result = qa_pipeline({
                "question": user_question,
                "context": context
            })
            answer = result["answer"]
        except Exception as e:
            answer = "Sorry, I couldn't find an answer. Try rephrasing your question."

    # Show assistant's answer
    with st.chat_message("assistant"):
        st.write(answer)

    st.session_state.chat.append({"role": "assistant", "content": answer})
