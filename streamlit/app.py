import streamlit as st
from model import HRLLMModel
import base64

st.set_page_config(page_title="HR Chat Assistant", page_icon="🟠", layout="wide")

if 'model' not in st.session_state:
    st.session_state.model = HRLLMModel()
if 'is_generating' not in st.session_state:
    st.session_state.is_generating = False
if 'response_text' not in st.session_state:
    st.session_state.response_text = ""
if 'query' not in st.session_state:
    st.session_state.query = ""
if 'knowledge_base' not in st.session_state:
    st.session_state.knowledge_base = "Hubs"
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'model_choice' not in st.session_state:
    available_models = st.session_state.model.get_available_models()
    st.session_state.model_choice = available_models[0] if available_models else "llama3"

def add_logo(logo_path):
    """
    Adds a logo to the sidebar at the top.
    The logo will have transparency where white appears in the original PNG.
    """
    try:
        with open(logo_path, "rb") as img_file:
            encoded_string = base64.b64encode(img_file.read()).decode()
        
        logo_html = f"""
        <div style="display: flex; justify-content: center; margin-bottom: 20px;">
            <img src="data:image/png;base64,{encoded_string}" alt="Logo" style="max-width: 100%;">
        </div>
        """
        st.sidebar.markdown(logo_html, unsafe_allow_html=True)
    except Exception as e:
        st.sidebar.warning(f"Unable to load logo: {e}")

st.markdown(
    """
    <style>
        body { background-color: #FF6200; color: white; }
        .stButton>button { background-color: #FF6200; color: white; border-radius: 5px; }
        .chat-message { padding: 10px; border-radius: 5px; margin-bottom: 10px; }
        .user-message { background-color: #FF8533; }
        .assistant-message { background-color: #CC4F00; }
    </style>
    """,
    unsafe_allow_html=True
)

try:
    add_logo("streamlit/images/big_logo.png")
except Exception as e:
    st.sidebar.warning(f"Unable to load logo: {e}")


def on_model_change():
    prev_model = st.session_state.get("previous_model", None)
    new_model = st.session_state.model_choice

    if prev_model and prev_model != new_model:
        st.session_state.chat_history.append({
            "role": "system",
            "content": f"🔄 Model switched from **{prev_model}** to **{new_model}**."
        })

    st.session_state.model.select_model(new_model)
    st.session_state.previous_model = new_model

def on_input_change():
    st.session_state.query = st.session_state.input_value


def on_faq_select():
    if st.session_state.faq_choice:
        st.session_state.query = st.session_state.faq_choice
        st.session_state.input_value = st.session_state.faq_choice


def send_message():
    if st.session_state.query:
        st.session_state.is_generating = True
        st.session_state.response_text = ""

        st.session_state.chat_history.append({"role": "user", "content": st.session_state.query})

        st.session_state.input_value = ""


def cancel_generation():
    st.session_state.is_generating = False

st.sidebar.title("Settings")

available_models = st.session_state.model.get_available_models()
default_index = 0
if st.session_state.model_choice in available_models:
    default_index = available_models.index(st.session_state.model_choice)


prev_model = st.session_state.get("previous_model", None)
new_model = st.session_state.model_choice



st.sidebar.selectbox(
    "Choose a model:", 
    available_models,
    index=default_index,
    key="model_choice",
    on_change=on_model_change
)

st.sidebar.selectbox(
    "Select Knowledge Base:", 
    ["Hubs", "NL", "RO"],
    index=["Hubs", "NL", "RO"].index(st.session_state.knowledge_base),
    key="knowledge_base"
)

faq_options = ["", "How do I request leave?", "What are the company policies?", "How do I update my details?", 
               "Where can I find my payslip?", "Who do I contact for HR support?"]
st.sidebar.selectbox(
    "Frequently Asked Questions:", 
    faq_options,
    key="faq_choice",
    on_change=on_faq_select
)

if st.session_state.model_choice:
    st.session_state.model.select_model(st.session_state.model_choice)

st.title("🟠 HR LLM Chat Assistant")

for message in st.session_state.chat_history:
    if message["role"] == "user":
        st.markdown(f"<div class='chat-message user-message'><b>You:</b> {message['content']}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='chat-message assistant-message'><b>Assistant:</b> {message['content']}</div>", unsafe_allow_html=True)

st.text_input(
    "Ask me an HR-related question:", 
    value=st.session_state.query,
    key="input_value",
    on_change=on_input_change,
    disabled=st.session_state.is_generating
)


col1, col2 = st.columns([1, 1])

with col1:
    send_button = st.button("Send", on_click=send_message, disabled=st.session_state.is_generating or not st.session_state.query)

with col2:
    cancel_button = st.button("Cancel", on_click=cancel_generation, disabled=not st.session_state.is_generating)

if st.session_state.is_generating:
    st.markdown("<b>Assistant:</b>", unsafe_allow_html=True)
    message_placeholder = st.empty()
    
    for word in st.session_state.model.generate_response(st.session_state.query):
        if not st.session_state.is_generating:
            break
        
        st.session_state.response_text += word
        message_placeholder.markdown(st.session_state.response_text)
    
    if st.session_state.response_text and st.session_state.is_generating:
        st.session_state.chat_history.append({"role": "assistant", "content": st.session_state.response_text})
    
    st.session_state.is_generating = False
    st.session_state.query = ""
    
    st.rerun()