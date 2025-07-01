# 🩺 Health Chatbot – AI-Powered Health Assistant (Q&A Style)

**Health Chatbot** is a simple chatbot built using Streamlit and Hugging Face Transformers. It allows users to ask health-related questions and receive accurate answers based on a custom knowledge base (`context.txt`). No API key, no cloud backend, no setup hassle.

## 🌐 Live App  
👉 [Click here to use Health Chatbot](https://rishiy7-health-chatbot.streamlit.app/)

---

### ✨ Features

- 🧠 Uses Hugging Face's `question-answering` pipeline (no API needed)
- 📄 Powered by a custom `context.txt` with 15+ real-world health Q&A pairs
- 💬 Chat-style interface built with Streamlit
- 🚀 Fast and lightweight — runs on CPU, no GPU required
- 🌐 Easy to deploy on **Streamlit Cloud**
- 🔐 No login, API key, or internet model hosting required

---

### 📁 Files in This Project

| File | Description |
|------|-------------|
| `app.py` | Main Streamlit chatbot interface |
| `context.txt` | Health knowledge base (Q&A style) |
| `requirements.txt` | Required Python libraries |

---

### ▶️ How to Run Locally

```bash
git clone https://github.com/RishiY7/Health-Chatbot
cd Health-Chatbot
pip install -r requirements.txt
streamlit run app.py



