import streamlit as st
from transformers import pipeline

MODEL_NAME = "distilgpt2"
MAX_RESPONSE_LENGTH = 200
DEFAULT_PROMPT = "Ask me a health question..."

@st.cache_resource
def load_model():
    """Load the distilgpt2 model using Hugging Face pipeline."""
    return pipeline("text-generation", model=MODEL_NAME)

class HealthBot:
    def __init__(self):
        self.model = load_model()
        self.init_chat_history()

    def init_chat_history(self):
        if "messages" not in st.session_state:
            st.session_state.messages = []

    def generate_response(self, prompt: str) -> str:
        if not prompt.strip():
            return "Please enter a valid question."
        try:
            response = self.model(
                prompt,
                max_length=MAX_RESPONSE_LENGTH,
                do_sample=True,
                top_k=50,
                truncation=True
            )[0]['generated_text']
            return response.strip()
        except Exception as e:
            return f"Model error: {e}"

    def run(self):
        st.set_page_config(
            page_title="🩺 HealthBot",
            layout="wide",
            page_icon="🩺"
        )
        st.title("🩺 Health Chatbot")
        st.markdown("Ask your health-related questions below.")

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

# Run the app
if __name__ == "__main__":
    bot = HealthBot()
    bot.run()
