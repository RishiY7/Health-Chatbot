import streamlit as st
from transformers import pipeline

# Constants
DEFAULT_PROMPT = "Ask me a health question..."

@st.cache_resource
def load_qa_pipeline():
    return pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

@st.cache_data
def load_context():
    with open("context.txt", "r") as f:
        return f.read()

class HealthBot:
    def __init__(self):
        self.qa_pipeline = load_qa_pipeline()
        self.context = load_context()
        self.init_chat_history()

    def init_chat_history(self):
        if "messages" not in st.session_state:
            st.session_state.messages = []

    def generate_response(self, question: str) -> str:
        if not question.strip():
            return "Please enter a valid question."
        try:
            result = self.qa_pipeline(question=question, context=self.context)
            return result["answer"]
        except Exception as e:
            return f"Error generating answer: {e}"

    def run(self):
        st.set_page_config(
            page_title="🩺 HealthBot",
            layout="wide",
            page_icon="🩺"
        )
        st.title("🩺 Health Chatbot")
        st.markdown("Ask any health-related question. Note: For educational use only!")

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        if prompt := st.chat_input(DEFAULT_PROMPT):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.write(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = self.generate_response(prompt)
                st.write(response)

            st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    bot = HealthBot()
    bot.run()
