# app.py

# HealthBot - Question Answering Health Assistant by Rishi Yadav

import streamlit as st
from transformers import pipeline

# Load health Q&A context
with open("context.txt", "r", encoding="utf-8") as f:
    context = f.read()

# Load QA model
qa = pipeline("question-answering")

# Streamlit UI setup
st.set_page_config(page_title="🩺 Health Chatbot", layout="wide")
st.title("🩺 Health Chatbot")
st.markdown("Ask any general health question. The bot will answer using trusted health info.")

# Chat history
if "chat" not in st.session_state:
    st.session_state.chat = []

# Display past messages
for msg in st.session_state.chat:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Response function
def get_answer(question):
    if len(question.strip()) < 5:
        return "Please ask a more complete question."
    try:
        result = qa(question=question, context=context)
        return result["answer"]
    except Exception as e:
        return "Sorry, I couldn't find an answer."

# Take user input
user_q = st.chat_input("Enter your health question...")

if user_q:
    st.session_state.chat.append({"role": "user", "content": user_q})
    with st.chat_message("user"):
        st.write(user_q)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            reply = get_answer(user_q)
            st.write(reply)
        st.session_state.chat.append({"role": "assistant", "content": reply})
