import streamlit as st
from transformers import pipeline

# Constants
DEFAULT_PROMPT = "Ask me a health question..."

# Load QA model (cached to avoid reloading every time)
@st.cache_resource
def load_qa_pipeline():
    return pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

# Sample context (you can expand this or customize it)
CONTEXT = """
I am a health assistant designed to give general health guidance.
Common symptoms like fever, cough, cold, fatigue, and headache may indicate infections or lifestyle issues.
Maintaining a balanced diet, exercise, hydration, and regular medical checkups is important.
Always consult a doctor for a real diagnosis or emergency care.
"""

class HealthBot:
    def __init__(self):
        self.qa_pipeline = load_qa_pipeline()
        self.init_chat_history()

    def init_chat_history(self):
        if "messages" not in st.session_state:
            st.session_state.messages = []

    def generate_response(self, question: str) -> str:
        if not question.strip():
            return "Please enter a valid question."
        try:
            result = self.qa_pipeline(question=question, context=CONTEXT)
            return result["answer"]
        except Exception as e:
            return f"Error generating answer: {e}"

    def run(self):
        st.set_page_config(
            page_title="🩺 HealthBot Q&A",
            layout="wide",
            page_icon="🩺"
        )
        st.title("🩺 HealthBot - Ask a Health Question")
        st.markdown("This bot uses a Q&A model trained on medical texts for educational responses only. ✅")

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

# Run the chatbot
if __name__ == "__main__":
    bot = HealthBot()
    bot.run()
