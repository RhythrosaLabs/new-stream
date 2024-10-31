# config.py

import streamlit as st

def get_api_keys():
    """
    Retrieve API keys from the Streamlit sidebar.
    """
    with st.sidebar:
        st.header("🔑 API Keys")
        openai_api_key = st.text_input("OpenAI API Key", type="password")
        stability_api_key = st.text_input("Stability AI API Key", type="password")
        st.markdown("[Get an OpenAI API key](https://platform.openai.com/account/api-keys)")
        st.markdown("[Get a Stability AI API key](https://platform.stability.ai/account/api-keys)")
        st.markdown("---")
        uploaded_file = st.file_uploader("📄 Upload a document for Q&A", type=["txt", "pdf", "md"])
    
    return openai_api_key, stability_api_key, uploaded_file
