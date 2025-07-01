# app.py

# HealthBot - Simple Health Chat App using Transformers and Streamlit
# Created by Rishi Yadav

import streamlit as st
from transformers import pipeline

# Load health context
with open("context.txt", "r", encoding="utf-8") as f:
    context = f.read()

# Load question-answering model
qa_pipeline = pipeline("question-answering")

# Streamlit app config
st.set_page_config(page_title="🩺 HealthBot", layout="wide")
st.title("🩺 HealthBot Assistant")
st.markdown("Ask any general health-related question and get instant guidance!")

# Store chat history
if "chat" not in st.session_state:
    st.session_state.chat = []

# Show previous messages
for msg in st.session_state.chat:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# User input
user_question = st.chat_input("Enter your health question...")

if user_question:
    st.session_state.chat.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.write(user_question)

    # Simple input validation
    if len(user_question.strip().split()) < 2:
        answer = "Can you please ask a more specific health-related question?"
    else:
        with st.spinner("Thinking..."):
            try:
                result = qa_pipeline({
                    "question": user_question,
                    "context": context
                })
                answer = result["answer"]
            except Exception as e:
                answer = "Sorry, I couldn't find an answer. Please try rephrasing your question."

    with st.chat_message("assistant"):
        st.write(answer)

    st.session_state.chat.append({"role": "assistant", "content": answer})
