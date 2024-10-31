import streamlit as st
from openai import OpenAI
import anthropic

# Page config and title
st.set_page_config(page_title="AI Assistant Hub", layout="wide")
st.title("🤖 AI Assistant Hub")

# Sidebar for API keys and navigation
with st.sidebar:
    st.header("API Configuration")
    openai_api_key = st.text_input("OpenAI API Key", key="openai_api_key", type="password")
    anthropic_api_key = st.text_input("Anthropic API Key", key="anthropic_api_key", type="password")
    
    st.markdown("---")
    st.markdown("""
    - [Get OpenAI API key](https://platform.openai.com/account/api-keys)
    - [Get Anthropic API key](https://console.anthropic.com/)
    """)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I'm your AI assistant. You can:\n1. Chat with GPT-3.5\n2. Ask questions about uploaded files\n\nHow can I help you today?"}]

# Create tabs for different functionalities
tab1, tab2 = st.tabs(["💬 Chat", "📝 File Q&A"])

# Tab 1: Basic Chat with GPT-3.5
with tab1:
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])
    
    if chat_prompt := st.chat_input("Send a message"):
        if not openai_api_key:
            st.info("Please add your OpenAI API key to continue.")
            st.stop()
            
        try:
            client = OpenAI(api_key=openai_api_key)
            st.session_state.messages.append({"role": "user", "content": chat_prompt})
            st.chat_message("user").write(chat_prompt)
            
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=st.session_state.messages
            )
            msg = response.choices[0].message.content
            st.session_state.messages.append({"role": "assistant", "content": msg})
            st.chat_message("assistant").write(msg)
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

# Tab 2: File Q&A with Claude
with tab2:
    uploaded_file = st.file_uploader("Upload an article", type=("txt", "md"))
    question = st.text_input(
        "Ask something about the article",
        placeholder="Can you give me a short summary?",
        disabled=not uploaded_file,
    )
    
    if uploaded_file and question:
        if not anthropic_api_key:
            st.info("Please add your Anthropic API key to continue.")
            st.stop()
            
        try:
            article = uploaded_file.read().decode()
            client = anthropic.Client(api_key=anthropic_api_key)
            
            message = client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=1000,
                messages=[{
                    "role": "user",
                    "content": f"Here's an article:\n\n{article}\n\n{question}"
                }]
            )
            
            st.write("### Answer")
            st.write(message.content[0].text)
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
