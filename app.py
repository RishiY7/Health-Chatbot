# app.py

# 🩺 HealthBot by Rishi Yadav
# A simple health assistant chatbot using GPT-2 and Streamlit

import streamlit as st
from transformers import pipeline, set_seed

# Load health-related knowledge from file
with open("context.txt", "r", encoding="utf-8") as file:
    context = file.read()

# Load GPT-2 text generation model
generator = pipeline("text-generation", model="gpt2", tokenizer="gpt2")
set_seed(42)  # To make answers slightly more consistent

# Streamlit app setup
st.set_page_config(page_title="🩺 HealthBot", layout="wide")
st.title("🩺 HealthBot")
st.markdown("Ask any general health question and get detailed guidance instantly!")

# Keep chat history
if "chat" not in st.session_state:
    st.session_state.chat = []

# Show past messages
for msg in st.session_state.chat:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Function to generate a response using context and GPT-2
def generate_answer(question):
    # Combine context and user question
    prompt = f"{context}\nUser: {question}\nAssistant:"
    try:
        output = generator(
            prompt,
            max_length=300,
            temperature=0.9,
            top_k=50,
            top_p=0.95,
            do_sample=True,
            num_return_sequences=1
        )
        # Extract just the response part
        full_text = output[0]["generated_text"]
        answer = full_text.split("Assistant:")[-1].strip()
        return answer
    except Exception as e:
        return f"Sorry, there was a problem generating the response.\n\nError: {e}"

# Take input from the user
user_question = st.chat_input("Enter your health question...")

# If question is entered, respond
if user_question:
    st.session_state.chat.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.write(user_question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            reply = generate_answer(user_question)
        st.write(reply)
    st.session_state.chat.append({"role": "assistant", "content": reply})
